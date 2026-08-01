# Agent Evaluation Metrics

Evaluating an AI agent requires more than measuring whether it produces a correct final answer. An agent may need to select tools, create a plan, execute multiple steps, recover from errors, control costs, and complete a task safely.

A practical evaluation framework should therefore measure five areas:

1. **Task success**
2. **Answer quality**
3. **Tool-use quality**
4. **Efficiency**
5. **Safety and reliability**

---

## 1. Task Success Rate

Task Success Rate measures how often the agent successfully completes the assigned task.

[
\text{Task Success Rate} =
\frac{\text{Successful Tasks}}{\text{Total Tasks}}
]

A task may be evaluated automatically, manually, or through a combination of both.

```python
evaluation_results = [
    {"task_id": 1, "success": True},
    {"task_id": 2, "success": False},
    {"task_id": 3, "success": True},
    {"task_id": 4, "success": True},
]

successful_tasks = sum(result["success"] for result in evaluation_results)
total_tasks = len(evaluation_results)

success_rate = successful_tasks / total_tasks

print(f"Task success rate: {success_rate:.2%}")
```

Example output:

```text
Task success rate: 75.00%
```

### When to use it

Task Success Rate is useful when tasks have clearly defined completion conditions, such as:

* Successfully booking an appointment
* Retrieving the correct customer record
* Generating a valid report
* Resolving a support request
* Updating the correct database entry

---

## 2. Exact Match Accuracy

Exact Match Accuracy measures whether the agent's answer exactly matches the expected answer.

[
\text{Exact Match Accuracy} =
\frac{\text{Exact Matches}}{\text{Total Responses}}
]

```python
expected_answers = [
    "Paris",
    "42",
    "Approved",
]

agent_answers = [
    "Paris",
    "42",
    "Pending",
]

matches = [
    expected.strip().lower() == actual.strip().lower()
    for expected, actual in zip(expected_answers, agent_answers)
]

exact_match_accuracy = sum(matches) / len(matches)

print(f"Exact match accuracy: {exact_match_accuracy:.2%}")
```

Exact matching is appropriate for structured tasks with one clearly correct answer. It is usually too strict for open-ended tasks.

---

## 3. Semantic Similarity

Semantic Similarity measures how closely the meaning of an agent's answer matches a reference answer.

This metric is useful when multiple answers may express the same idea using different wording.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

reference_answer = (
    "Customers can reset their password through the account settings page."
)

agent_answer = (
    "Users may change a forgotten password from their account settings."
)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([reference_answer, agent_answer])

similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

print(f"Semantic similarity: {similarity:.3f}")
```

For production-grade evaluation, embedding models are generally more effective than TF-IDF because they capture meaning rather than only word overlap.

---

## 4. Precision, Recall, and F1 Score

These metrics are useful when an agent must identify multiple relevant items, such as:

* Relevant documents
* Security risks
* Customer complaints
* Required actions
* Entities in a request

### Precision

Precision measures how many selected items were actually relevant.

[
\text{Precision} =
\frac{TP}{TP + FP}
]

### Recall

Recall measures how many relevant items were successfully identified.

[
\text{Recall} =
\frac{TP}{TP + FN}
]

### F1 Score

F1 Score balances precision and recall.

[
F1 =
2 \times
\frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
]

```python
expected_items = {
    "data leakage",
    "prompt injection",
    "unauthorized access",
}

identified_items = {
    "data leakage",
    "prompt injection",
    "slow response time",
}

true_positives = len(expected_items & identified_items)
false_positives = len(identified_items - expected_items)
false_negatives = len(expected_items - identified_items)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1_score:.2f}")
```

---

## 5. Tool Selection Accuracy

Tool Selection Accuracy measures whether the agent selected the correct tool for a task.

For example:

* Using a weather service for current weather
* Using a calculator for arithmetic
* Using a database search tool for customer information
* Avoiding unnecessary tools when the answer is already available

[
\text{Tool Selection Accuracy} =
\frac{\text{Correct Tool Decisions}}{\text{Total Tool Decisions}}
]

```python
tool_evaluations = [
    {"expected": "calculator", "selected": "calculator"},
    {"expected": "database_search", "selected": "web_search"},
    {"expected": "weather_api", "selected": "weather_api"},
    {"expected": None, "selected": None},
]

correct_decisions = sum(
    item["expected"] == item["selected"]
    for item in tool_evaluations
)

accuracy = correct_decisions / len(tool_evaluations)

print(f"Tool selection accuracy: {accuracy:.2%}")
```

This metric should also consider whether the agent correctly decided **not** to use a tool.

---

## 6. Tool Execution Success Rate

An agent may select the correct tool but still use it incorrectly. Tool Execution Success Rate measures whether tool calls are valid and successfully completed.

[
\text{Tool Execution Success Rate} =
\frac{\text{Successful Tool Calls}}{\text{Total Tool Calls}}
]

```python
tool_calls = [
    {"tool": "search", "status": "success"},
    {"tool": "calculator", "status": "success"},
    {"tool": "database", "status": "error"},
    {"tool": "email", "status": "success"},
]

successful_calls = sum(
    call["status"] == "success"
    for call in tool_calls
)

execution_success_rate = successful_calls / len(tool_calls)

print(f"Tool execution success rate: {execution_success_rate:.2%}")
```

Common execution problems include:

* Missing required arguments
* Invalid parameter formats
* Incorrect identifiers
* Unauthorized operations
* Repeated tool calls
* Misinterpretation of returned data

---

## 7. Tool Argument Accuracy

Tool Argument Accuracy evaluates whether the agent supplies the correct parameters to a tool.

```python
expected_arguments = {
    "location": "Istanbul",
    "date": "2026-08-03",
}

actual_arguments = {
    "location": "Istanbul",
    "date": "2026-08-04",
}

correct_fields = sum(
    actual_arguments.get(key) == value
    for key, value in expected_arguments.items()
)

argument_accuracy = correct_fields / len(expected_arguments)

print(f"Tool argument accuracy: {argument_accuracy:.2%}")
```

Field-level accuracy is especially useful for evaluating agents that work with APIs, forms, databases, or business applications.

---

## 8. Trajectory Accuracy

Trajectory Accuracy evaluates the complete sequence of actions taken by an agent.

Consider an expected trajectory:

```text
Search customer → Open account → Verify eligibility → Apply discount
```

The agent may reach the correct final result through an unnecessarily long or incorrect path. Trajectory evaluation captures this difference.

```python
expected_trajectory = [
    "search_customer",
    "open_account",
    "verify_eligibility",
    "apply_discount",
]

actual_trajectory = [
    "search_customer",
    "open_account",
    "verify_eligibility",
    "apply_discount",
]

trajectory_match = expected_trajectory == actual_trajectory

print(f"Exact trajectory match: {trajectory_match}")
```

An exact comparison may be too strict when multiple valid paths exist. A step-level score can be used instead.

```python
matching_steps = sum(
    expected == actual
    for expected, actual in zip(expected_trajectory, actual_trajectory)
)

step_accuracy = matching_steps / len(expected_trajectory)

print(f"Trajectory step accuracy: {step_accuracy:.2%}")
```

---

## 9. Step Efficiency

Step Efficiency compares the number of actions taken by the agent with the expected or minimum number of actions.

[
\text{Step Efficiency} =
\frac{\text{Minimum Required Steps}}
{\text{Actual Steps}}
]

A score of `1.0` means the agent used the expected minimum number of steps.

```python
minimum_required_steps = 4
actual_steps = 6

step_efficiency = minimum_required_steps / actual_steps

print(f"Step efficiency: {step_efficiency:.2f}")
```

Step Efficiency should not be optimized independently. An agent that skips necessary verification steps may appear efficient but create additional risk.

---

## 10. Redundant Action Rate

Redundant Action Rate measures the percentage of actions that did not contribute meaningfully to task completion.

[
\text{Redundant Action Rate} =
\frac{\text{Redundant Actions}}{\text{Total Actions}}
]

```python
actions = [
    {"name": "search_customer", "redundant": False},
    {"name": "open_customer", "redundant": False},
    {"name": "search_customer_again", "redundant": True},
    {"name": "update_record", "redundant": False},
]

redundant_actions = sum(action["redundant"] for action in actions)
redundancy_rate = redundant_actions / len(actions)

print(f"Redundant action rate: {redundancy_rate:.2%}")
```

Lower values are better.

---

## 11. Latency

Latency measures the time required for the agent to complete a task.

Common latency metrics include:

* Mean latency
* Median latency
* P95 latency
* P99 latency
* Time to first response
* Total task completion time

```python
import numpy as np

latencies = [1.2, 1.8, 2.1, 1.5, 4.7, 2.3, 1.9]

mean_latency = np.mean(latencies)
median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)

print(f"Mean latency: {mean_latency:.2f} seconds")
print(f"Median latency: {median_latency:.2f} seconds")
print(f"P95 latency: {p95_latency:.2f} seconds")
```

Median latency is often more representative than mean latency when a small number of tasks take unusually long.

---

## 12. Cost per Task

Cost per Task measures the average computational or financial cost required to complete a task.

The calculation may include:

* Input token cost
* Output token cost
* Tool-call cost
* Infrastructure cost
* External API cost

```python
tasks = [
    {"input_cost": 0.012, "output_cost": 0.018, "tool_cost": 0.005},
    {"input_cost": 0.010, "output_cost": 0.015, "tool_cost": 0.000},
    {"input_cost": 0.020, "output_cost": 0.025, "tool_cost": 0.010},
]

total_cost = sum(
    task["input_cost"] + task["output_cost"] + task["tool_cost"]
    for task in tasks
)

average_cost = total_cost / len(tasks)

print(f"Total cost: ${total_cost:.3f}")
print(f"Average cost per task: ${average_cost:.3f}")
```

Cost should be considered together with quality. A cheaper agent is not necessarily better if it completes fewer tasks successfully.

---

## 13. Token Efficiency

Token Efficiency measures how effectively the agent uses its available context and generated tokens.

One simple formulation is:

[
\text{Token Efficiency} =
\frac{\text{Successful Tasks}}
{\text{Total Tokens Used}}
]

```python
runs = [
    {"success": True, "tokens": 1200},
    {"success": True, "tokens": 900},
    {"success": False, "tokens": 1800},
]

successful_tasks = sum(run["success"] for run in runs)
total_tokens = sum(run["tokens"] for run in runs)

successes_per_1000_tokens = (
    successful_tasks / total_tokens
) * 1000

print(
    "Successful tasks per 1,000 tokens: "
    f"{successes_per_1000_tokens:.2f}"
)
```

This metric is mainly useful for comparing agents that perform similar tasks.

---

## 14. Groundedness

Groundedness measures whether the agent's claims are supported by the information available in its context or retrieved sources.

A basic claim-level groundedness score can be calculated as:

[
\text{Groundedness} =
\frac{\text{Supported Claims}}
{\text{Total Verifiable Claims}}
]

```python
claims = [
    {"text": "The customer placed the order on Monday.", "supported": True},
    {"text": "The order contains two products.", "supported": True},
    {"text": "The customer requested express delivery.", "supported": False},
]

supported_claims = sum(claim["supported"] for claim in claims)
groundedness = supported_claims / len(claims)

print(f"Groundedness: {groundedness:.2%}")
```

Groundedness evaluation may be performed through:

* Rule-based source matching
* Human evaluation
* Citation verification
* A separate evaluator model

---

## 15. Hallucination Rate

Hallucination Rate measures the proportion of generated claims that are unsupported or factually incorrect.

[
\text{Hallucination Rate} =
\frac{\text{Unsupported Claims}}
{\text{Total Verifiable Claims}}
]

```python
total_verifiable_claims = 12
unsupported_claims = 2

hallucination_rate = unsupported_claims / total_verifiable_claims

print(f"Hallucination rate: {hallucination_rate:.2%}")
```

Lower values are better.

A hallucination metric should distinguish between:

* Unsupported claims
* Factually incorrect claims
* Outdated claims
* Misinterpreted source content
* Fabricated citations

---

## 16. Citation Accuracy

Citation Accuracy measures whether each citation actually supports the claim associated with it.

```python
citation_evaluations = [
    {"citation_id": "source_1", "supports_claim": True},
    {"citation_id": "source_2", "supports_claim": False},
    {"citation_id": "source_3", "supports_claim": True},
]

supported_citations = sum(
    item["supports_claim"]
    for item in citation_evaluations
)

citation_accuracy = supported_citations / len(citation_evaluations)

print(f"Citation accuracy: {citation_accuracy:.2%}")
```

Citation evaluation should check:

1. Whether the source exists
2. Whether the cited passage is relevant
3. Whether the source supports the complete claim
4. Whether the source is sufficiently current and reliable

---

## 17. Constraint Compliance

Constraint Compliance measures whether the agent follows task-specific instructions.

Examples include:

* Required output format
* Word limit
* Allowed tools
* Prohibited actions
* Required fields
* Language requirements
* Business rules

```python
constraints = {
    "uses_markdown": True,
    "under_word_limit": True,
    "contains_required_sections": True,
    "contains_prohibited_content": False,
}

passed_constraints = [
    constraints["uses_markdown"],
    constraints["under_word_limit"],
    constraints["contains_required_sections"],
    not constraints["contains_prohibited_content"],
]

compliance_score = sum(passed_constraints) / len(passed_constraints)

print(f"Constraint compliance: {compliance_score:.2%}")
```

For critical constraints, a binary pass/fail evaluation may be more appropriate than an average score.

---

## 18. Safety Violation Rate

Safety Violation Rate measures how often an agent performs an unsafe, unauthorized, or prohibited action.

[
\text{Safety Violation Rate} =
\frac{\text{Unsafe Runs}}
{\text{Total Runs}}
]

```python
runs = [
    {"run_id": 1, "safety_violation": False},
    {"run_id": 2, "safety_violation": False},
    {"run_id": 3, "safety_violation": True},
    {"run_id": 4, "safety_violation": False},
]

violations = sum(run["safety_violation"] for run in runs)
violation_rate = violations / len(runs)

print(f"Safety violation rate: {violation_rate:.2%}")
```

Examples of safety violations include:

* Exposing private data
* Performing unauthorized transactions
* Ignoring approval requirements
* Executing harmful instructions
* Bypassing access controls
* Sending information to the wrong recipient

---

## 19. Human Escalation Accuracy

Agents should recognize situations that require human review.

Human Escalation Accuracy evaluates whether the agent escalates when necessary and avoids unnecessary escalation.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# 1 means escalation is required.
expected_escalation = [1, 0, 1, 0, 1, 0]

# 1 means the agent escalated.
agent_escalation = [1, 0, 0, 0, 1, 1]

precision = precision_score(expected_escalation, agent_escalation)
recall = recall_score(expected_escalation, agent_escalation)
f1 = f1_score(expected_escalation, agent_escalation)

print(f"Escalation precision: {precision:.2f}")
print(f"Escalation recall: {recall:.2f}")
print(f"Escalation F1: {f1:.2f}")
```

High recall may be particularly important in high-risk domains because failing to escalate a critical case can have serious consequences.

---

## 20. Recovery Rate

Recovery Rate measures whether an agent can recover after an error, failed tool call, or unexpected result.

[
\text{Recovery Rate} =
\frac{\text{Successfully Recovered Errors}}
{\text{Recoverable Errors}}
]

```python
error_events = [
    {"recoverable": True, "recovered": True},
    {"recoverable": True, "recovered": False},
    {"recoverable": False, "recovered": False},
    {"recoverable": True, "recovered": True},
]

recoverable_errors = [
    event for event in error_events
    if event["recoverable"]
]

successful_recoveries = sum(
    event["recovered"]
    for event in recoverable_errors
)

recovery_rate = (
    successful_recoveries / len(recoverable_errors)
    if recoverable_errors
    else 0
)

print(f"Recovery rate: {recovery_rate:.2%}")
```

Recovery behavior may include:

* Correcting invalid tool arguments
* Trying an alternative data source
* Asking for required missing information
* Revising an incorrect plan
* Stopping safely after repeated failures

---

## 21. Consistency

Consistency measures whether repeated runs produce similar levels of quality.

```python
import statistics

scores = [0.89, 0.91, 0.87, 0.90, 0.72]

mean_score = statistics.mean(scores)
standard_deviation = statistics.stdev(scores)

print(f"Mean quality score: {mean_score:.3f}")
print(f"Score standard deviation: {standard_deviation:.3f}")
```

A lower standard deviation generally indicates more consistent performance.

However, repeated outputs do not need to be identical. The objective is stable quality and behavior, not identical wording.

---

## 22. Composite Agent Score

A composite score combines several metrics into a single summary value.

For example:

[
\text{Agent Score} =
0.40(\text{Task Success}) +
0.20(\text{Groundedness}) +
0.15(\text{Tool Accuracy}) +
0.15(\text{Efficiency}) +
0.10(\text{Safety})
]

```python
metrics = {
    "task_success": 0.86,
    "groundedness": 0.92,
    "tool_accuracy": 0.80,
    "efficiency": 0.74,
    "safety": 0.98,
}

weights = {
    "task_success": 0.40,
    "groundedness": 0.20,
    "tool_accuracy": 0.15,
    "efficiency": 0.15,
    "safety": 0.10,
}

composite_score = sum(
    metrics[name] * weights[name]
    for name in metrics
)

print(f"Composite agent score: {composite_score:.3f}")
```

Composite scores are useful for dashboards, but they can hide important weaknesses. Individual metrics should always remain visible.

For example, a strong average score should not compensate for a serious privacy or safety failure.

---

## 23. Example Evaluation Dataset

An agent evaluation dataset should contain the expected behavior for each test case.

```python
evaluation_cases = [
    {
        "case_id": "case_001",
        "input": "Calculate the total price of three products.",
        "expected_tool": "calculator",
        "expected_result": 125.50,
        "requires_escalation": False,
        "max_steps": 3,
    },
    {
        "case_id": "case_002",
        "input": "Refund a transaction above the approval limit.",
        "expected_tool": "payment_system",
        "expected_result": "human_approval_required",
        "requires_escalation": True,
        "max_steps": 4,
    },
]
```

Each case may include:

* User input
* Context
* Expected final result
* Expected tool or tools
* Allowed alternative tools
* Required intermediate checks
* Maximum number of steps
* Safety constraints
* Escalation requirements
* Evaluation rubric

---

## 24. Simple Evaluation Function

The following example evaluates success, tool selection, step efficiency, and safety.

```python
def evaluate_agent_run(
    expected_result,
    actual_result,
    expected_tool,
    selected_tool,
    maximum_steps,
    actual_steps,
    safety_violation,
):
    task_success = int(actual_result == expected_result)
    tool_accuracy = int(selected_tool == expected_tool)

    step_efficiency = min(
        maximum_steps / max(actual_steps, 1),
        1.0,
    )

    safety_score = 0 if safety_violation else 1

    composite_score = (
        0.50 * task_success
        + 0.20 * tool_accuracy
        + 0.15 * step_efficiency
        + 0.15 * safety_score
    )

    return {
        "task_success": task_success,
        "tool_accuracy": tool_accuracy,
        "step_efficiency": round(step_efficiency, 3),
        "safety_score": safety_score,
        "composite_score": round(composite_score, 3),
    }


result = evaluate_agent_run(
    expected_result="approved",
    actual_result="approved",
    expected_tool="approval_system",
    selected_tool="approval_system",
    maximum_steps=4,
    actual_steps=5,
    safety_violation=False,
)

print(result)
```

Example output:

```python
{
    "task_success": 1,
    "tool_accuracy": 1,
    "step_efficiency": 0.8,
    "safety_score": 1,
    "composite_score": 0.97,
}
```

---

## 25. Evaluation Dashboard Example

The following example creates a summary table from multiple agent runs.

```python
import pandas as pd

runs = [
    {
        "run_id": 1,
        "task_success": 1,
        "groundedness": 0.95,
        "tool_accuracy": 1,
        "latency_seconds": 2.4,
        "cost_usd": 0.04,
        "safety_violation": 0,
    },
    {
        "run_id": 2,
        "task_success": 0,
        "groundedness": 0.60,
        "tool_accuracy": 0,
        "latency_seconds": 5.8,
        "cost_usd": 0.09,
        "safety_violation": 0,
    },
    {
        "run_id": 3,
        "task_success": 1,
        "groundedness": 0.90,
        "tool_accuracy": 1,
        "latency_seconds": 3.1,
        "cost_usd": 0.05,
        "safety_violation": 0,
    },
]

df = pd.DataFrame(runs)

summary = {
    "task_success_rate": df["task_success"].mean(),
    "average_groundedness": df["groundedness"].mean(),
    "tool_selection_accuracy": df["tool_accuracy"].mean(),
    "average_latency_seconds": df["latency_seconds"].mean(),
    "average_cost_usd": df["cost_usd"].mean(),
    "safety_violation_rate": df["safety_violation"].mean(),
}

for metric, value in summary.items():
    print(f"{metric}: {value:.3f}")
```

---

## 26. Recommended Metric Set

A balanced agent evaluation scorecard should include at least the following metrics:

| Dimension         | Recommended Metric                   |
| ----------------- | ------------------------------------ |
| Outcome           | Task Success Rate                    |
| Answer quality    | Correctness or Semantic Similarity   |
| Retrieval quality | Groundedness and Citation Accuracy   |
| Tool use          | Tool Selection and Execution Success |
| Process           | Trajectory Accuracy                  |
| Efficiency        | Steps, Latency, Tokens, and Cost     |
| Reliability       | Consistency and Recovery Rate        |
| Governance        | Constraint Compliance                |
| Safety            | Safety Violation Rate                |
| Human oversight   | Escalation Precision and Recall      |

---

## 27. Evaluation Best Practices

### Use representative test cases

The evaluation dataset should reflect real user requests, edge cases, ambiguous inputs, and failure scenarios.

### Evaluate both outcomes and processes

A correct answer reached through unsafe or unauthorized actions should not be considered a successful run.

### Separate critical and non-critical metrics

Safety, privacy, and authorization requirements should often be treated as mandatory gates rather than weighted metrics.

### Test multiple runs

Agent behavior may vary between runs. Each important test case should be executed multiple times.

### Include adversarial cases

Test how the agent responds to:

* Prompt injection
* Conflicting instructions
* Missing information
* Tool failures
* Incorrect retrieved information
* Unauthorized requests
* Ambiguous user intent

### Combine automated and human evaluation

Automated metrics provide scale and consistency. Human reviewers are still useful for evaluating reasoning quality, relevance, clarity, and business appropriateness.

---

## Conclusion

Agent evaluation is a multidimensional problem. A strong evaluation system should not focus only on final-answer accuracy.

It should measure whether the agent:

* Completes the intended task
* Selects and uses tools correctly
* Follows an appropriate sequence of actions
* Produces grounded and reliable answers
* Operates efficiently
* Recovers from failures
* Complies with constraints
* Escalates high-risk cases
* Avoids unsafe or unauthorized behavior

The most appropriate metrics depend on the agent's purpose, operating environment, and level of autonomy. High-risk agents should place greater emphasis on safety, authorization, traceability, and human oversight, while high-volume service agents may also prioritize latency, cost, and completion rate.
