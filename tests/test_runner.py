from __future__ import annotations

from core.plan import (
    ExecutionResult,
    FeedbackDecision,
    FeedbackOutcome,
    Plan,
    StaticPlanner,
    validate_plan,
)
from core.runner import Runner, _collect_plan_skills
from core.skills import Skill, SkillRegistry


class ArmGrabMockSkill(Skill):
    name = "arm.grab"
    description = "mock arm grab"
    parameters = []
    returns = "str"

    def __init__(self, returns: list[str]) -> None:
        self._returns = returns
        self._idx = 0

    def __call__(self, **kwargs: object) -> str:
        if self._idx >= len(self._returns):
            return self._returns[-1]
        value = self._returns[self._idx]
        self._idx += 1
        return value


class MockSkill(Skill):
    name = "mock.skill"
    description = "mock skill"
    parameters = []
    returns = "Any"

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value

    def __call__(self, **kwargs: object) -> object:
        return self.return_value


class ListObjectsSkill(Skill):
    name = "state.list_objects"
    description = "list objects"
    parameters = []
    returns = "list"

    def __init__(self, objects: list[str]) -> None:
        self._objects = objects

    def __call__(self, **kwargs: object) -> list[str]:
        return self._objects


class FakeBackend:
    def __init__(self, success: bool = True, objects: list[str] | None = None) -> None:
        self._success = success
        self._objects = objects or ["obj_a", "obj_b"]
        self.closed = False
        self.register_calls: list[tuple[str, str]] = []

    def register_skills(self, registry: SkillRegistry, goal: str) -> None:
        self.register_calls.append(("register", goal))
        registry.register(MockSkill("ok"))
        registry.register(ListObjectsSkill(self._objects))

    def capture_image(self) -> bytes:
        return b"img"

    def success(self) -> bool:
        return self._success

    def close(self) -> None:
        self.closed = True


class MockPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self._plans = plans
        self.calls: list[tuple[str, str, str]] = []

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        self.calls.append((goal, catalog, context))
        if self._plans:
            return self._plans.pop(0)
        raise RuntimeError("No plan available")


class SequenceFeedbackEvaluator:
    def __init__(self, decisions: list[FeedbackDecision]) -> None:
        self._decisions = decisions
        self._idx = 0

    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackOutcome:
        if self._idx >= len(self._decisions):
            return FeedbackOutcome(decision=self._decisions[-1])
        decision = self._decisions[self._idx]
        self._idx += 1
        return FeedbackOutcome(decision=decision)


def make_runner(
    plans: list[Plan],
    decisions: list[FeedbackDecision],
    step_budget: int = 30,
    max_replans: int = 3,
    trace_dir: str | None = None,
    backend_success: bool = True,
) -> tuple[FakeBackend, MockPlanner, Runner, SequenceFeedbackEvaluator]:
    backend = FakeBackend(success=backend_success)
    planner = MockPlanner(plans)
    feedback = SequenceFeedbackEvaluator(decisions)
    runner = Runner(max_replans=max_replans, step_budget=step_budget, trace_dir=trace_dir)
    return backend, planner, runner, feedback


def handcrafted_plan() -> Plan:
    return validate_plan(
        {
            "goal": "pick up the cube",
            "body": {
                "kind": "seq",
                "steps": [
                    {
                        "kind": "call",
                        "skill": "arm.grab",
                        "args": {"object_name": {"kind": "literal", "value": "cube"}},
                        "assign_to": "grab_result",
                    },
                    {"kind": "feedback", "description": "Is the cube grasped?"},
                    {"kind": "return", "value": {"kind": "var", "name": "grab_result"}},
                ],
            },
        }
    )


def test_runner_done() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "call", "skill": "mock.skill"}})
    backend, planner, runner, feedback = make_runner([plan], [FeedbackDecision.CONTINUE])
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "done"
    assert result.success is True
    assert result.value is None
    assert backend.closed is True


def test_runner_return_value() -> None:
    plan = Plan.model_validate(
        {"goal": "g", "body": {"kind": "return", "value": {"kind": "literal", "value": 99}}}
    )
    backend, planner, runner, feedback = make_runner([plan], [FeedbackDecision.CONTINUE])
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "done"
    assert result.value == 99


def test_runner_stopped() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "stop now"}})
    backend, planner, runner, feedback = make_runner([plan], [FeedbackDecision.STOP])
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "stopped"
    assert result.success is False


def test_runner_replan_then_done() -> None:
    first = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    second = Plan.model_validate({"goal": "g", "body": {"kind": "call", "skill": "mock.skill"}})
    backend, planner, runner, feedback = make_runner(
        [first, second], [FeedbackDecision.REPLAN, FeedbackDecision.CONTINUE]
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "done"
    assert result.replans == 1
    assert len([c for c in planner.calls if c[2]]) == 1


def test_runner_replan_limit() -> None:
    first = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "feedback", "description": "retry"},
                ],
            },
        }
    )
    second = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry-again"}})
    backend, planner, runner, feedback = make_runner(
        [first, second], [FeedbackDecision.REPLAN, FeedbackDecision.REPLAN], max_replans=0
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "replan_limit"
    assert result.steps == 1
    assert result.replans == 0


def test_runner_replan_limit_trace_steps(tmp_path) -> None:
    import json

    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "feedback", "description": "retry"},
                ],
            },
        }
    )
    backend, planner, runner, feedback = make_runner(
        [plan], [FeedbackDecision.REPLAN], max_replans=0, trace_dir=str(tmp_path)
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "replan_limit"
    assert result.steps == 1

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    events = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    plan_end = next(e for e in events if e["event"] == "plan_end")
    assert plan_end["steps"] == 1


class InitialThenFailingPlanner:
    def __init__(self, initial: Plan) -> None:
        self._initial = initial

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        if not context:
            return self._initial
        raise RuntimeError("LLMPlanner: plan validation failed after 3 attempts")


def test_runner_replan_planner_failure() -> None:
    first = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    backend = FakeBackend()
    runner = Runner(max_replans=3)
    result = runner.run(
        backend,
        "g",
        planner=InitialThenFailingPlanner(first),
        feedback=SequenceFeedbackEvaluator([FeedbackDecision.REPLAN]),
    )
    assert result.status == "replan_failed"
    assert result.replans == 1
    assert "plan validation failed" in (result.error or "")


def test_runner_step_budget() -> None:
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "call", "skill": "mock.skill"},
                ],
            },
        }
    )
    backend, planner, runner, feedback = make_runner(
        [plan], [FeedbackDecision.CONTINUE], step_budget=2
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "step_budget"


def test_collect_plan_skills() -> None:
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "perception.detect"},
                    {"kind": "call", "skill": "arm.grab"},
                    {"kind": "feedback", "description": "grasped?"},
                    {"kind": "call", "skill": "arm.place"},
                ],
            },
        }
    )
    assert _collect_plan_skills(plan.body) == [
        "perception.detect",
        "arm.grab",
        "arm.place",
    ]


def test_plan_start_trace_includes_skills(tmp_path) -> None:
    import json

    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "feedback", "description": "halt"},
                    {"kind": "call", "skill": "perception.detect"},
                    {"kind": "call", "skill": "arm.grab"},
                    {"kind": "call", "skill": "arm.place"},
                ],
            },
        }
    )
    backend, planner, runner, feedback = make_runner(
        [plan], [FeedbackDecision.STOP], trace_dir=str(tmp_path)
    )
    runner.run(backend, "g", planner=planner, feedback=feedback)

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    first_event = json.loads(trace_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert first_event["event"] == "plan_start"
    assert first_event["skills"] == [
        "perception.detect",
        "arm.grab",
        "arm.place",
    ]


def test_happy_path_continue() -> None:
    plan = handcrafted_plan()
    backend = FakeBackend()
    backend.register_skills = lambda reg, goal: reg.register(ArmGrabMockSkill(["grabbed"]))
    runner = Runner()
    result = runner.run(
        backend,
        "pick up the cube",
        planner=StaticPlanner(plan),
        feedback=SequenceFeedbackEvaluator([FeedbackDecision.CONTINUE]),
    )
    assert result.status == "done"
    assert result.replans == 0


def test_replan_path() -> None:
    plan = handcrafted_plan()
    backend = FakeBackend()
    backend.register_skills = lambda reg, goal: reg.register(
        ArmGrabMockSkill(["failed", "grabbed"])
    )
    runner = Runner(max_replans=1)
    result = runner.run(
        backend,
        "pick up the cube",
        planner=StaticPlanner(plan),
        feedback=SequenceFeedbackEvaluator(
            [FeedbackDecision.REPLAN, FeedbackDecision.CONTINUE]
        ),
    )
    assert result.replans == 1
    assert result.status != "replan_limit"


def test_verdict_fails_when_status_not_done() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "stop"}})
    backend, planner, runner, feedback = make_runner([plan], [FeedbackDecision.STOP])
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.success is False


def test_verdict_fails_on_failure_return_value() -> None:
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {"kind": "literal", "value": "failed to grasp object"},
            },
        }
    )
    backend, planner, runner, feedback = make_runner(
        [plan], [FeedbackDecision.CONTINUE], backend_success=True
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "done"
    assert result.success is False


def test_verdict_succeeds_when_done_and_backend_success() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "call", "skill": "mock.skill"}})
    backend, planner, runner, feedback = make_runner(
        [plan], [FeedbackDecision.CONTINUE], backend_success=True
    )
    result = runner.run(backend, "g", planner=planner, feedback=feedback)
    assert result.status == "done"
    assert result.success is True
