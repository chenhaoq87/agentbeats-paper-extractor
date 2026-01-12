"""Green agent for evaluating dairy science paper extraction."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import contextlib
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TaskState,
    TextPart,
    DataPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from helpers import (
    load_paper_xml,
    load_gold_output,
    build_participant_payload,
    evaluate_response,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dairy_paper_evaluator")


class BERTScore(BaseModel):
    """BERTScore metrics."""
    precision: float
    recall: float
    f1: float


class PaperEvaluation(BaseModel):
    """Evaluation results for a single paper."""
    equation_match_percentage: float
    bertscore: BERTScore
    error: Optional[str] = None


class PaperResult(BaseModel):
    """Result for a single paper evaluation."""
    paper_id: str
    evaluation: Optional[PaperEvaluation] = None
    prediction: Optional[str] = None
    gold_output: Optional[str] = None
    error: Optional[str] = None


class OverallResults(BaseModel):
    """Overall evaluation results across all papers."""
    overall_score: float
    mean_equation_match_percentage: float
    mean_bertscore_f1: float
    total_papers: int
    successful_evaluations: int
    papers: list[PaperResult]


def paper_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the paper evaluator."""
    skill = AgentSkill(
        id="evaluate-papers",
        name="Evaluate Papers",
        description="Evaluates a participant on a set of dairy science papers.",
        tags=["evaluation", "benchmark"],
        examples=[
            """{
  "participants": {"participant": "http://remote-agent"},
  "config": {}
}"""
        ],
    )

    return AgentCard(
        name=name,
        description="A Green Agent that evaluates participant agents on dairy science paper extraction tasks.",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


class PaperEvaluator(GreenAgent):
    """
    Green agent for evaluating dairy science paper extraction.

    This agent:
    - Processes papers from the data directory
    - Sends papers to participants for extraction
    - Evaluates responses against gold outputs
    - Provides comprehensive evaluation metrics
    """

    def __init__(self):
        self._required_roles = ["participant"]
        self._tool_provider = ToolProvider()
        # Set data directory paths (can be overridden via environment variables)
        base_dir = Path(__file__).parent
        self.input_dir = Path(
            os.getenv("INPUT_DIR", str(base_dir / "data" / "input"))
        )
        self.output_dir = Path(
            os.getenv("OUTPUT_DIR", str(base_dir / "data" / "output"))
        )
        self.templates_dir = Path(
            os.getenv("TEMPLATES_DIR", str(base_dir / "templates"))
        )
        # Define paper IDs list
        self.paper_ids = ["001", "002", "003", "004"]
        # Timeout for participant requests (in seconds, default 5 minutes)
        self.timeout = int(os.getenv("PARTICIPANT_TIMEOUT", "300"))

        logger.info(
            f"Initialized agent with input_dir={self.input_dir}, output_dir={self.output_dir}"
        )

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the evaluation request.

        Args:
            request: The evaluation request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        return True, "ok"

    def validate_paper_files(self) -> tuple[bool, str]:
        """Validate that all required paper files exist.

        Returns:
            Tuple of (all_exist, error_message)
        """
        missing_files = []
        for paper_id in self.paper_ids:
            xml_file = self.input_dir / f"Paper{paper_id}_raw.xml"
            json_file = self.output_dir / f"Paper{paper_id}_answer.json"
            if not xml_file.exists():
                missing_files.append(f"Input: {xml_file}")
            if not json_file.exists():
                missing_files.append(f"Output: {json_file}")

        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        return True, "ok"

    async def process_single_paper(
        self, paper_id: str, participant_url: str, updater: TaskUpdater
    ) -> PaperResult:
        """Process a single paper: load, send to participant, and evaluate.

        Args:
            paper_id: The paper ID to process
            participant_url: URL of the participant agent
            updater: Task updater for status updates

        Returns:
            PaperResult with evaluation results
        """
        logger.info(f"Processing Paper{paper_id}")

        try:
            # Load XML content
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading Paper{paper_id} XML..."),
            )
            xml_content = load_paper_xml(paper_id, self.input_dir)
            logger.info(f"Loaded XML for Paper{paper_id} ({len(xml_content)} chars)")

            # Build payload and send to participant
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Sending Paper{paper_id} to participant..."),
            )
            payload = build_participant_payload(xml_content, self.templates_dir)

            # Use ToolProvider to communicate with participant
            # Note: ToolProvider doesn't support timeout parameter, so we'll use default
            prediction = await self._tool_provider.talk_to_agent(
                message=payload,
                url=str(participant_url),
                new_conversation=True,  # isolate each paper
            )
            logger.info(
                f"Received response for Paper{paper_id} ({len(prediction)} chars)"
            )

            # Load gold output
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating Paper{paper_id} response..."),
            )
            gold_output = load_gold_output(paper_id, self.output_dir)

            # Evaluate response
            evaluation_dict = evaluate_response(
                prediction=prediction,
                gold_output=gold_output,
                lang="en",
            )

            prediction_equations = evaluation_dict.get("prediction_equations", [])
            gold_equations = evaluation_dict.get("gold_equations", [])

            # Build readable equation comparison message
            paired_lines = []
            max_len = max(len(prediction_equations), len(gold_equations))
            for idx in range(max_len):
                pred_eq = (
                    prediction_equations[idx]
                    if idx < len(prediction_equations)
                    else "<missing>"
                )
                gold_eq = gold_equations[idx] if idx < len(gold_equations) else "<missing>"
                paired_lines.append(
                    f"[{idx + 1}] extracted: {pred_eq}\n    gold: {gold_eq}"
                )

            eq_log_message = (
                f"Paper{paper_id} equations:\n" + "\n".join(paired_lines)
                if paired_lines
                else f"Paper{paper_id} equations: none found"
            )

            # Convert evaluation dict to PaperEvaluation model
            if "error" in evaluation_dict:
                evaluation = PaperEvaluation(
                    equation_match_percentage=0.0,
                    bertscore=BERTScore(precision=0.0, recall=0.0, f1=0.0),
                    error=evaluation_dict["error"],
                )
            else:
                bertscore_dict = evaluation_dict.get("bertscore", {})
                evaluation = PaperEvaluation(
                    equation_match_percentage=evaluation_dict.get(
                        "equation_match_percentage", 0.0
                    ),
                    bertscore=BERTScore(
                        precision=bertscore_dict.get("precision", 0.0),
                        recall=bertscore_dict.get("recall", 0.0),
                        f1=bertscore_dict.get("f1", 0.0),
                    ),
                )

            logger.info(eq_log_message)

            score = evaluation.equation_match_percentage
            logger.info(
                f"Paper{paper_id} evaluation complete: "
                f"equation_match={score:.2f}%, bertscore_f1={evaluation.bertscore.f1:.4f}"
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(eq_log_message),
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Paper{paper_id} evaluation complete: "
                    f"equation_match={score:.2f}%, bertscore_f1={evaluation.bertscore.f1:.4f}"
                ),
            )

            return PaperResult(
                paper_id=paper_id,
                evaluation=evaluation,
                prediction=prediction,
                gold_output=gold_output,
            )

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            logger.error(f"Paper{paper_id} error: {error_msg}")
            logger.debug(traceback.format_exc())
            return PaperResult(paper_id=paper_id, error=error_msg)
        except Exception as e:
            error_msg = f"Error processing paper: {str(e)}"
            logger.error(f"Paper{paper_id} error: {error_msg}")
            logger.debug(traceback.format_exc())
            return PaperResult(paper_id=paper_id, error=error_msg)

    def calculate_overall_metrics(
        self, paper_results: list[PaperResult]
    ) -> OverallResults:
        """Calculate overall metrics from paper results.

        Args:
            paper_results: List of paper evaluation results

        Returns:
            OverallResults with aggregated metrics
        """
        successful_results = [
            r for r in paper_results if r.evaluation is not None and r.error is None
        ]

        total_papers = len(self.paper_ids)

        if successful_results:
            mean_eq_match = sum(
                r.evaluation.equation_match_percentage for r in successful_results
            ) / len(successful_results)

            mean_bertscore_f1 = sum(
                r.evaluation.bertscore.f1 for r in successful_results
            ) / len(successful_results)

            overall_score = (mean_eq_match / 100.0 + mean_bertscore_f1) / 2.0
        else:
            mean_eq_match = 0.0
            mean_bertscore_f1 = 0.0
            overall_score = 0.0

        return OverallResults(
            overall_score=overall_score,
            mean_equation_match_percentage=mean_eq_match,
            mean_bertscore_f1=mean_bertscore_f1,
            total_papers=total_papers,
            successful_evaluations=len(successful_results),
            papers=paper_results,
        )

    def serialize_results_without_content(self, overall_results: OverallResults) -> dict:
        """Serialize OverallResults excluding prediction and gold_output fields.
        
        Args:
            overall_results: OverallResults object to serialize
            
        Returns:
            Dictionary with evaluation metrics only, excluding agent input/output content
        """
        result_dict = overall_results.model_dump()
        # Remove prediction and gold_output from each paper result
        for paper in result_dict.get('papers', []):
            paper.pop('prediction', None)
            paper.pop('gold_output', None)
        return result_dict

    async def process_papers(
        self, participant_url: str, updater: TaskUpdater
    ) -> list[PaperResult]:
        """Process all papers sequentially.

        Args:
            participant_url: URL of the participant agent
            updater: Task updater for status updates

        Returns:
            List of PaperResult objects
        """
        paper_results: list[PaperResult] = []
        total_papers = len(self.paper_ids)

        logger.info(f"Starting evaluation of {total_papers} papers")

        for idx, paper_id in enumerate(self.paper_ids):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Evaluating Paper{paper_id} ({idx+1}/{total_papers})..."
                ),
            )

            result = await self.process_single_paper(
                paper_id, participant_url, updater
            )
            paper_results.append(result)

        logger.info(f"Completed evaluation of {total_papers} papers")
        return paper_results

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Main entry point for the agent evaluation.

        Args:
            req: The evaluation request
            updater: Report progress (update_status) and results (add_artifact)
        """
        logger.info(f"Starting evaluation with request: {req.model_dump_json()}")

        # Validate paper files exist
        ok, msg = self.validate_paper_files()
        if not ok:
            logger.error(f"Paper file validation failed: {msg}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(msg),
            )
            return

        participant_url = str(req.participants["participant"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting paper evaluation.\n{req.model_dump_json()}"
            ),
        )

        # Process all papers
        paper_results = await self.process_papers(participant_url, updater)

        # Calculate overall metrics
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Paper processing finished. Calculating overall metrics..."
            ),
        )
        logger.info("Paper processing finished. Calculating overall metrics.")

        overall_results = self.calculate_overall_metrics(paper_results)
        # Log only evaluation metrics, excluding prediction and gold_output content
        results_summary = self.serialize_results_without_content(overall_results)
        logger.info(f"Overall Results:\n{json.dumps(results_summary, indent=2)}")

        # Create result summary text
        result_text = (
            f"Overall score: {overall_results.overall_score:.4f}\n"
            f"Mean equation match: {overall_results.mean_equation_match_percentage:.2f}%\n"
            f"Mean BERTScore F1: {overall_results.mean_bertscore_f1:.4f}\n"
            f"Successful evaluations: {overall_results.successful_evaluations}/{overall_results.total_papers}"
        )

        # Create EvalResult for compatibility
        # Exclude prediction and gold_output from artifact to avoid printing agent input/output
        results_data = self.serialize_results_without_content(overall_results)
        result = EvalResult(
            winner="participant", detail=results_data
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data=results_data)),
            ],
            name="Result",
        )

        # Reset tool provider context
        self._tool_provider.reset()


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A paper evaluator.")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9009, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url",
    )
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel

        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(
            args.card_url or f"http://{args.host}:{args.port}/"
        )

    async with agent_url_cm as agent_url:
        agent = PaperEvaluator()
        executor = GreenExecutor(agent)
        agent_card = paper_evaluator_agent_card("PaperEvaluator", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())

