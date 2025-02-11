#!/usr/bin/env python3

import click
from pathlib import Path
import os
import sys
import json
from typing import Dict, List, Optional
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
import time
from openai import OpenAI
from anthropic import Anthropic
import asyncio
import time
import logging
from dotenv import dotenv_values

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    print(f"Error: .env file not found at {env_path}")
    sys.exit(1)

# Убедимся, что логгер настроен
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_extractor")

# Initialize clients and validate API keys

config = dotenv_values(env_path)
openai_api_key = config.get("OPENAI_API_KEY")
anthropic_api_key = config.get("ANTHROPIC_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in .env file")
    sys.exit(1)

if not anthropic_api_key:
    logger.error("ANTHROPIC_API_KEY not found in .env file")
    sys.exit(1)

try:
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    
    # Verify API clients are working
    logger.info("Testing API clients...")
    
    # Test OpenAI client
    test_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test message"}],
        max_tokens=10
    )
    logger.info("OpenAI client initialized successfully")
    
    # Test Anthropic client
    test_response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=10,
        messages=[{"role": "user", "content": "Test message"}]
    )
    logger.info("Anthropic client initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing API clients: {str(e)}", exc_info=True)
    if hasattr(e, 'response'):
        logger.error(f"API Response: {e.response}")
    sys.exit(1)

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger("project_extractor")

def generate_tree(path: Path, prefix: str = "", max_depth: int = 4, current_depth: int = 0) -> List[str]:
    """Generate a tree structure of the directory up to max_depth"""
    EXCLUDED_DIRS = {
        'node_modules', '.next', '.git', 'venv', '__pycache__', 
        'dist', 'build', '.vscode', '.idea', 'coverage',
        '.pytest_cache', '.mypy_cache', 'env', '.env', '.venv',
        'site-packages'
    }
    
    EXCLUDED_FILES = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        '.DS_Store', '.env', '.env.local', '.gitignore',
        'README.md', 'LICENSE', '.eslintrc', '.prettierrc',
        'tsconfig.json', 'requirements.txt', 'poetry.lock',
        'Pipfile.lock'
    }
    
    EXCLUDED_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.ico',
        '.svg', '.mp4', '.mp3', '.pdf', '.zip',
        '.woff', '.woff2', '.ttf', '.eot',
        '.pyc', '.pyo', '.pyd', '.so', '.pkl', '.pickle',
        '.db', '.sqlite', '.log', '.cache'
    }
    
    if current_depth >= max_depth:
        return ["    " + prefix + "..."]
    
    try:
        items = sorted(path.glob('*'))
        filtered_items = []
        
        for item in items:
            if item.is_dir():
                if item.name not in EXCLUDED_DIRS and not item.name.startswith('.'):
                    filtered_items.append(item)
            elif item.is_file():
                if (item.name not in EXCLUDED_FILES and 
                    item.suffix.lower() not in EXCLUDED_EXTENSIONS and
                    not item.name.startswith('.')):
                    filtered_items.append(item)
        
        tree = []
        for i, item in enumerate(filtered_items):
            is_last = i == len(filtered_items) - 1
            current_prefix = "└── " if is_last else "├── "
            item_name = f"{item.name}/" if item.is_dir() else item.name
            tree.append(f"{prefix}{current_prefix}{item_name}")
            
            if item.is_dir():
                extension = "    " if is_last else "│   "
                tree.extend(generate_tree(
                    item,
                    prefix + extension,
                    max_depth,
                    current_depth + 1
                ))
        
        return tree
    except Exception as e:
        logger.error(f"Error processing {path}: {str(e)}")
        return [f"{prefix}└── <e>"]

class ClaudeAgent:
    def __init__(self, name: str, role: str, responsibilities: List[str]):
        self.name = name
        self.role = role
        self.responsibilities = responsibilities
        
    async def analyze(self, context: Dict) -> Dict:
        """Run agent analysis using claude-3-5-sonnet-latest"""
        try:
            full_response = ""
            current_content = ""
            
            with Live(Panel("Analyzing...", title=f"{self.name} Analysis", border_style="blue"), refresh_per_second=4) as live:
                with anthropic_client.messages.stream(
                    max_tokens=8000,
                    messages=[{
                        "role": "user",
                        "content": f"""You are the {self.name}, responsible for {self.role}.
                        
                        Your specific responsibilities are:
                        {chr(10).join(f'- {r}' for r in self.responsibilities)}
                        
                        Analyze this project context and provide a detailed report focused on your domain:
                        
                        {json.dumps(context, indent=2)}
                        
                        Format your response as a structured report with clear sections and findings."""
                    }],
                    model="claude-3-5-sonnet-latest"
                ) as stream:
                    for event in stream:
                        if event.type == "message_start":
                            live.update(Panel("Starting analysis...", title=f"{self.name} Analysis", border_style="blue"))
                        elif event.type == "content_block_start":
                            current_content = ""
                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                text = event.delta.text
                                current_content += text
                                full_response += text
                                live.update(Panel(current_content, title=f"{self.name} Analysis", border_style="blue"))
                        elif event.type == "content_block_stop":
                            pass
                        elif event.type == "message_delta":
                            if event.delta.stop_reason:
                                live.update(Panel("Analysis complete!", title=f"{self.name} Analysis", border_style="green"))
                        elif event.type == "message_stop":
                            pass
                        elif event.type == "error":
                            error_msg = f"Error: {event.error.message}"
                            live.update(Panel(error_msg, title="Error", border_style="red"))
                            raise Exception(error_msg)
            
            return {
                "agent": self.name,
                "findings": full_response
            }
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {str(e)}", exc_info=True)
            if hasattr(e, 'response'):
                logger.error(f"API Response: {e.response}")
            return {
                "agent": self.name,
                "error": str(e)
            }

async def run_openai_completion(model: str, messages: List[Dict], title: str) -> Dict:
    """Run OpenAI completion with streaming"""
    try:
        full_response = ""
        current_content = ""
        
        # Используем глобальный клиент, который уже инициализирован с API ключом
        global openai_client
        
        with Live(Panel("Processing...", title=title, border_style="blue"), refresh_per_second=4) as live:
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    current_content += text
                    full_response += text
                    live.update(Panel(current_content, title=title, border_style="blue"))
            
            live.update(Panel("Complete!", title=title, border_style="green"))
        
        return {
            "content": full_response,
            "tokens": len(full_response.split())  # Rough estimate
        }
    except Exception as e:
        logger.error(f"Error in OpenAI completion: {str(e)}", exc_info=True)
        if hasattr(e, 'response'):
            logger.error(f"API Response: {e.response}")
        return {
            "error": str(e)
        }

class ProjectAnalyzer:
    def __init__(self, directory: Path):
        self.directory = directory
        
        # Initialize result storage
        self.phase1_results = {}
        self.phase2_results = {}
        self.phase3_results = {}
        self.phase4_results = {}
        self.consolidated_report = {}
        self.final_analysis = {}
        
        # Phase 1: Initial Discovery Agents
        self.phase1_agents = [
            ClaudeAgent("Structure Agent", "analyzing directory and file organization", [
                "Analyze directory and file organization",
                "Map project layout and file relationships",
                "Identify key architectural components"
            ]),
            ClaudeAgent("Dependency Agent", "investigating packages and libraries", [
                "Investigate all packages and libraries",
                "Determine version requirements",
                "Research compatibility issues"
            ]),
            ClaudeAgent("Tech Stack Agent", "identifying frameworks and technologies", [
                "Identify all frameworks and technologies",
                "Gather latest documentation for each",
                "Note current best practices and updates"
            ])
        ]
        
        # Phase 3: Deep Analysis Agents
        self.phase3_agents = [
            ClaudeAgent("Code Analysis Agent", "examining core logic and patterns", [
                "Examine core logic and patterns",
                "Review implementation details",
                "Identify optimization opportunities"
            ]),
            ClaudeAgent("Dependency Mapping Agent", "mapping file relationships", [
                "Map all file relationships",
                "Document import/export patterns",
                "Chart data flow paths"
            ]),
            ClaudeAgent("Architecture Agent", "studying design patterns", [
                "Study design patterns",
                "Review architectural decisions",
                "Evaluate system structure"
            ]),
            ClaudeAgent("Documentation Agent", "creating comprehensive documentation", [
                "Create comprehensive docs",
                "Maintain analysis records",
                "Format findings clearly"
            ])
        ]
        
    async def run_phase1(self, tree: List[str], package_info: Dict) -> Dict:
        """Initial Discovery Phase using Claude-3.5-sonnet"""
        context = {
            "tree_structure": tree,
            "package_info": package_info
        }
        
        agent_tasks = [agent.analyze(context) for agent in self.phase1_agents]
        results = await asyncio.gather(*agent_tasks)
        
        return {
            "phase": "Initial Discovery",
            "findings": results
        }
        
    async def run_phase2(self, phase1_results: Dict) -> Dict:
        """Methodical Planning Phase using o3-mini"""
        try:
            logger.info("Starting Phase 2 with OpenAI API...")
            response = await run_openai_completion(
                model="o3-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Process these agent findings and create a detailed, step-by-step analysis plan:

                    Agent Findings:
                    {json.dumps(phase1_results, indent=2)}

                    Create a comprehensive plan including:
                    1. File-by-file examination approach
                    2. Critical areas needing investigation
                    3. Documentation requirements
                    4. Inter-dependency mapping method
                    """
                }],
                title="Phase 2: Methodical Planning"
            )
            
            if "error" in response:
                return {
                    "phase": "Methodical Planning",
                    "error": response["error"]
                }
            
            return {
                "phase": "Methodical Planning",
                "plan": response["content"],
                "reasoning_tokens": response["tokens"]
            }
        except Exception as e:
            logger.error(f"Error in Phase 2: {str(e)}")
            return {
                "phase": "Methodical Planning",
                "error": str(e)
            }
            
    async def run_phase3(self, analysis_plan: Dict, tree: List[str]) -> Dict:
        """Deep Analysis Phase using Claude-3.5-sonnet"""
        context = {
            "analysis_plan": analysis_plan,
            "tree_structure": tree
        }
        
        agent_tasks = [agent.analyze(context) for agent in self.phase3_agents]
        results = await asyncio.gather(*agent_tasks)
        
        return {
            "phase": "Deep Analysis",
            "findings": results
        }
        
    async def run_phase4(self, phase3_results: Dict) -> Dict:
        """Synthesis Phase using o3-mini"""
        try:
            logger.info("Starting Phase 4 with OpenAI API...")
            response = await run_openai_completion(
                model="o3-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Review and synthesize these agent findings:

                    Analysis Results:
                    {json.dumps(phase3_results, indent=2)}

                    Provide:
                    1. Deep analysis of all findings
                    2. Methodical processing of new information
                    3. Updated analysis directions
                    4. Refined instructions for agents
                    5. Areas needing deeper investigation
                    """
                }],
                title="Phase 4: Synthesis"
            )
            
            if "error" in response:
                return {
                    "phase": "Synthesis",
                    "error": response["error"]
                }
            
            return {
                "phase": "Synthesis",
                "analysis": response["content"],
                "reasoning_tokens": response["tokens"]
            }
        except Exception as e:
            logger.error(f"Error in Phase 4: {str(e)}")
            return {
                "phase": "Synthesis",
                "error": str(e)
            }
            
    async def run_phase5(self, all_results: Dict) -> Dict:
        """Consolidation Phase using Claude-3.5-sonnet"""
        try:
            full_response = ""
            current_content = ""
            
            with Live(Panel("Consolidating...", title="Phase 5: Consolidation", border_style="blue"), refresh_per_second=4) as live:
                with anthropic_client.messages.stream(
                    max_tokens=8000,
                    messages=[{
                        "role": "user",
                        "content": f"""As the Report Agent, create a comprehensive final report from all analysis phases:
                        
                        Analysis Results:
                        {json.dumps(all_results, indent=2)}
                        
                        Your tasks:
                        1. Combine all agent findings
                        2. Organize by component/module
                        3. Create comprehensive documentation
                        4. Highlight key discoveries
                        5. Prepare final report for O1"""
                    }],
                    model="claude-3-5-sonnet-latest"
                ) as stream:
                    for event in stream:
                        if event.type == "message_start":
                            live.update(Panel("Starting consolidation...", title="Phase 5: Consolidation", border_style="blue"))
                        elif event.type == "content_block_start":
                            current_content = ""
                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                text = event.delta.text
                                current_content += text
                                full_response += text
                                live.update(Panel(current_content, title="Phase 5: Consolidation", border_style="blue"))
                        elif event.type == "content_block_stop":
                            pass
                        elif event.type == "message_delta":
                            if event.delta.stop_reason:
                                live.update(Panel("Consolidation complete!", title="Phase 5: Consolidation", border_style="green"))
                        elif event.type == "message_stop":
                            pass
                        elif event.type == "error":
                            error_msg = f"Error: {event.error.message}"
                            live.update(Panel(error_msg, title="Error", border_style="red"))
                            raise Exception(error_msg)
            
            return {
                "phase": "Consolidation",
                "report": full_response
            }
        except Exception as e:
            logger.error(f"Error in Phase 5: {str(e)}", exc_info=True)
            if hasattr(e, 'response'):
                logger.error(f"API Response: {e.response}")
            return {
                "phase": "Consolidation",
                "error": str(e)
            }
            
    async def run_final_analysis(self, consolidated_report: Dict) -> Dict:
        """Final Analysis Phase using o3-mini"""
        try:
            logger.info("Starting Final Analysis with OpenAI API...")
            # Validate consolidated report
            if not consolidated_report or 'report' not in consolidated_report:
                logger.error("Invalid consolidated report: Missing or empty report")
                return {
                    "phase": "Final Analysis",
                    "error": "Invalid consolidated report structure"
                }
                
            response = await run_openai_completion(
                model="o3-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Process this consolidated report and provide a final analysis:

                    Consolidated Report:
                    {json.dumps(consolidated_report, indent=2)}

                    Provide:
                    1. Identified architectural patterns
                    2. Complete system structure mapping
                    3. Comprehensive relationship documentation
                    4. Improvement recommendations
                    5. Next analysis phase planning
                    """
                }],
                title="Final Analysis"
            )
            
            if "error" in response:
                return {
                    "phase": "Final Analysis",
                    "error": response["error"]
                }
            
            return {
                "phase": "Final Analysis",
                "analysis": response["content"],
                "reasoning_tokens": response["tokens"]
            }
        except Exception as e:
            logger.error(f"Error in Final Analysis: {str(e)}", exc_info=True)
            if hasattr(e, 'response'):
                logger.error(f"API Response: {e.response}")
            return {
                "phase": "Final Analysis",
                "error": str(e)
            }
            
    async def analyze(self) -> str:
        """Run complete analysis workflow"""
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Phase 1: Initial Discovery (Claude-3.5-sonnet)
            task1 = progress.add_task("[green]Phase 1: Initial Discovery...", total=None)
            tree = generate_tree(self.directory)
            package_info = {}  # You can implement package.json parsing here
            self.phase1_results = await self.run_phase1(tree, package_info)
            progress.update(task1, completed=True)
            
            # Phase 2: Methodical Planning (o3-mini)
            task2 = progress.add_task("[blue]Phase 2: Methodical Planning...", total=None)
            self.phase2_results = await self.run_phase2(self.phase1_results)
            progress.update(task2, completed=True)
            
            # Phase 3: Deep Analysis (Claude-3.5-sonnet)
            task3 = progress.add_task("[yellow]Phase 3: Deep Analysis...", total=None)
            self.phase3_results = await self.run_phase3(self.phase2_results, tree)
            progress.update(task3, completed=True)
            
            # Phase 4: Synthesis (o3-mini)
            task4 = progress.add_task("[magenta]Phase 4: Synthesis...", total=None)
            self.phase4_results = await self.run_phase4(self.phase3_results)
            progress.update(task4, completed=True)
            
            # Phase 5: Consolidation (Claude-3.5-sonnet)
            task5 = progress.add_task("[cyan]Phase 5: Consolidation...", total=None)
            all_results = {
                "phase1": self.phase1_results,
                "phase2": self.phase2_results,
                "phase3": self.phase3_results,
                "phase4": self.phase4_results
            }
            self.consolidated_report = await self.run_phase5(all_results)
            progress.update(task5, completed=True)
            
            # Final Analysis (o3-mini)
            task6 = progress.add_task("[white]Final Analysis...", total=None)
            self.final_analysis = await self.run_final_analysis(self.consolidated_report)
            progress.update(task6, completed=True)
            
            # New phase: Generate .cursorrules
            task7 = progress.add_task("[red]Generating .cursorrules file...", total=None)
            analysis_data = {
                "phase1": self.phase1_results,
                "phase2": self.phase2_results,
                "phase3": self.phase3_results,
                "phase4": self.phase4_results,
                "consolidated_report": self.consolidated_report,
                "final_analysis": self.final_analysis
            }
            generate_cursorrules(self.directory, analysis_data)
            progress.update(task7, completed=True)
        
        # Format final output
        analysis = [
            f"Project Analysis Report for: {self.directory}",
            "=" * 50 + "\n",
            "Phase 1: Initial Discovery (Claude-3.5-sonnet)",
            "-" * 30,
            json.dumps(self.phase1_results, indent=2),
            "\n",
            "Phase 2: Methodical Planning (o3-mini)",
            "-" * 30,
            self.phase2_results.get("plan", "Error in planning phase"),
            "\n",
            "Phase 3: Deep Analysis (Claude-3.5-sonnet)",
            "-" * 30,
            json.dumps(self.phase3_results, indent=2),
            "\n",
            "Phase 4: Synthesis (o3-mini)",
            "-" * 30,
            self.phase4_results.get("analysis", "Error in synthesis phase"),
            "\n",
            "Phase 5: Consolidation (Claude-3.5-sonnet)",
            "-" * 30,
            self.consolidated_report.get("report", "Error in consolidation phase"),
            "\n",
            "Final Analysis (o3-mini)",
            "-" * 30,
            self.final_analysis.get("analysis", "Error in final analysis phase"),
            "\n",
            "Analysis Metrics",
            "-" * 30,
            f"Time taken: {time.time() - start_time:.2f} seconds",
            f"Phase 2 reasoning tokens: {self.phase2_results.get('reasoning_tokens', 0)}",
            f"Phase 4 reasoning tokens: {self.phase4_results.get('reasoning_tokens', 0)}",
            f"Final Analysis reasoning tokens: {self.final_analysis.get('reasoning_tokens', 0)}"
        ]
        
        return "\n".join(analysis)

def save_phase_outputs(directory: Path, analysis_data: dict) -> None:
    """Save each phase's output to separate markdown files."""
    # Create phases_output directory
    output_dir = directory / "phases_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Initial Discovery
    with open(output_dir / "phase1_discovery.md", "w", encoding="utf-8") as f:
        f.write("# Phase 1: Initial Discovery (Claude-3.5-sonnet)\n\n")
        f.write("## Agent Findings\n\n")
        f.write("```json\n")
        f.write(json.dumps(analysis_data["phase1"], indent=2))
        f.write("\n```\n")
    
    # Phase 2: Methodical Planning
    with open(output_dir / "phase2_planning.md", "w", encoding="utf-8") as f:
        f.write("# Phase 2: Methodical Planning (o3-mini)\n\n")
        f.write(analysis_data["phase2"].get("plan", "Error in planning phase"))
    
    # Phase 3: Deep Analysis
    with open(output_dir / "phase3_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Phase 3: Deep Analysis (Claude-3.5-sonnet)\n\n")
        f.write("```json\n")
        f.write(json.dumps(analysis_data["phase3"], indent=2))
        f.write("\n```\n")
    
    # Phase 4: Synthesis
    with open(output_dir / "phase4_synthesis.md", "w", encoding="utf-8") as f:
        f.write("# Phase 4: Synthesis (o3-mini)\n\n")
        f.write(analysis_data["phase4"].get("analysis", "Error in synthesis phase"))
    
    # Phase 5: Consolidation
    with open(output_dir / "phase5_consolidation.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5: Consolidation (Claude-3.5-sonnet)\n\n")
        f.write(analysis_data["consolidated_report"].get("report", "Error in consolidation phase"))
    
    # Final Analysis
    with open(output_dir / "final_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Final Analysis (o3-mini)\n\n")
        f.write(analysis_data["final_analysis"].get("analysis", "Error in final analysis phase"))
    
    # Complete Report
    with open(output_dir / "complete_report.md", "w", encoding="utf-8") as f:
        f.write("# Complete Project Analysis Report\n\n")
        f.write(f"Project: {directory}\n")
        f.write("=" * 50 + "\n\n")
        f.write("## Analysis Metrics\n\n")
        f.write(f"- Time taken: {analysis_data['metrics']['time']:.2f} seconds\n")
        f.write(f"- Phase 2 reasoning tokens: {analysis_data['metrics']['phase2_tokens']}\n")
        f.write(f"- Phase 4 reasoning tokens: {analysis_data['metrics']['phase4_tokens']}\n")
        f.write(f"- Final Analysis reasoning tokens: {analysis_data['metrics']['final_tokens']}\n\n")
        f.write("See individual phase files for detailed outputs.")

def generate_cursorrules(directory: Path, analysis_data: dict) -> None:
    """Generate .cursorrules file based on project analysis and template."""
    try:
        # Read template
        template_path = directory / "cursorrules_template.txt"
        if not template_path.exists():
            logger.warning("Template file not found, using default sections")
            template_content = """# Instructions

During your interaction with the user, if you find anything reusable in this project (e.g. version of a library, model name), especially about a fix to a mistake you made or a correction you received, you should take note in the `Lessons` section in the `.cursorrules` file so you will not make the same mistake again. 

You should also use the `.cursorrules` file as a Scratchpad to organize your thoughts. Especially when you receive a new task, you should first review the content of the Scratchpad, clear old different task if necessary, first explain the task, and plan the steps you need to take to complete the task. You can use todo markers to indicate the progress, e.g.
[X] Task 1
[ ] Task 2

Also update the progress of the task in the Scratchpad when you finish a subtask.
Especially when you finished a milestone, it will help to improve your depth of task accomplishment to use the Scratchpad to reflect and plan.
The goal is to help you maintain a big picture as well as the progress of the task. Always refer to the Scratchpad when you plan the next step.

# Lessons

## User Specified Lessons

## Cursor learned

# Scratchpad
"""
        else:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

        # Extract findings from phase 1
        phase1_findings = analysis_data.get("phase1", {}).get("findings", [])
        
        # Initialize sections
        structure_section = []
        tools_section = []
        tech_stack_section = []
        infrastructure_section = []

        # Process each agent's findings
        for finding in phase1_findings:
            if isinstance(finding, dict):
                agent_type = finding.get("agent")
                findings = finding.get("findings", "")

                if agent_type == "Structure Agent":
                    if isinstance(findings, str):
                        # Extract Directory Organization
                        if "Directory Organization Overview" in findings:
                            dir_section = findings.split("Directory Organization Overview")[1].split("##")[0]
                            structure_section.extend([
                                "## Directory Organization",
                                *[line.strip() for line in dir_section.split("\n")
                                  if line.strip() and not line.strip().startswith("#")]
                            ])
                        
                        # Extract Key Components
                        if "Key Components Analysis" in findings:
                            components_section = findings.split("Key Components Analysis")[1].split("##")[0]
                            structure_section.extend([
                                "\n## Key Components",
                                *[line.strip() for line in components_section.split("\n")
                                  if line.strip() and not line.strip().startswith("#")]
                            ])
                        
                        # Extract Architectural Patterns
                        if "Architectural Patterns" in findings:
                            patterns_section = findings.split("Architectural Patterns")[1].split("##")[0]
                            structure_section.extend([
                                "\n## Architectural Patterns",
                                *[line.strip() for line in patterns_section.split("\n")
                                  if line.strip() and not line.strip().startswith("#")]
                            ])

                elif agent_type == "Tech Stack Agent":
                    if isinstance(findings, str):
                        # Extract Core Technologies
                        if "Core Technologies" in findings:
                            tech_section = findings.split("Core Technologies")[1].split("###")[0]
                            tech_stack_section.extend([
                                "## Core Technologies",
                                *[line.strip() for line in tech_section.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))]
                            ])
                        
                        # Extract Key Technical Components
                        if "Key Technical Components" in findings:
                            components = findings.split("Key Technical Components")[1].split("##")[0]
                            tech_stack_section.extend([
                                "\n## Technical Components",
                                *[line.strip() for line in components.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*") or line.strip().startswith("•"))]
                            ])
                        
                        # Extract Best Practices
                        if "Best Practices & Current Standards" in findings:
                            practices = findings.split("Best Practices & Current Standards")[1].split("##")[0]
                            tech_stack_section.extend([
                                "\n## Best Practices",
                                *[line.strip() for line in practices.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))]
                            ])

                elif agent_type == "Dependency Agent":
                    if isinstance(findings, str):
                        # Extract Required Core Dependencies
                        if "Required Core Dependencies" in findings:
                            deps_section = findings.split("Required Core Dependencies")[1].split("##")[0]
                            tools_section.extend([
                                "## Development Environment",
                                "### Required Dependencies",
                                *[line.strip() for line in deps_section.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))]
                            ])
                        
                        # Extract Optional Dependencies
                        if "Optional Dependencies" in findings:
                            opt_deps = findings.split("Optional Dependencies")[1].split("##")[0]
                            tools_section.extend([
                                "\n### Optional Dependencies",
                                *[line.strip() for line in opt_deps.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))]
                            ])
                        
                        # Extract Development Tools
                        if "Development Tools" in findings:
                            dev_tools = findings.split("Development Tools")[1].split("##")[0]
                            tools_section.extend([
                                "\n### Development Tools",
                                *[line.strip() for line in dev_tools.split("\n")
                                  if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))]
                            ])

        # Add infrastructure information from phase 3 or 4
        if "phase3" in analysis_data:
            phase3_findings = analysis_data["phase3"].get("findings", [])
            for finding in phase3_findings:
                if isinstance(finding, dict) and finding.get("agent") == "Architecture Agent":
                    arch_findings = finding.get("findings", "")
                    if isinstance(arch_findings, str):
                        infrastructure_section.extend([
                            "## System Architecture",
                            *[line.strip() for line in arch_findings.split("\n")
                              if line.strip() and line.strip().startswith("-")],
                            "\n## Implementation Patterns",
                            "- Agent Pattern: Specialized agents for specific responsibilities",
                            "- Repository Pattern: Clear directory structure",
                            "- Strategy Pattern: Different analysis capabilities",
                            "- Factory Pattern: Agent creation and management",
                            "- Observer Pattern: Event handling and monitoring",
                            "- Builder Pattern: Documentation generation"
                        ])

        # Extract structure information from phase 5
        phase5_report = analysis_data.get("consolidated_report", {}).get("report", "")
        
        if phase5_report and isinstance(phase5_report, str):
            # Extract Main Application Structure
            if "Main Application Structure" in phase5_report:
                structure_section.extend([
                    "## Project Overview",
                    "Main application components and organization:",
                    ""
                ])
                
                # Extract directory structure
                if "Directory Organization" in phase5_report:
                    dir_structure = phase5_report.split("```")[1].split("```")[0]
                    structure_section.extend([
                        "### Directory Structure",
                        "```",
                        dir_structure.strip(),
                        "```",
                        ""
                    ])

            # Extract Core Components
            if "Core Components Analysis" in phase5_report:
                structure_section.extend([
                    "### Core Components",
                    ""
                ])
                
                # Entry Point
                if "Entry Point (main.py)" in phase5_report:
                    entry_section = phase5_report.split("Entry Point (main.py)")[1].split("###")[0]
                    structure_section.extend([
                        "#### Entry Point (main.py)",
                        *[line.strip() for line in entry_section.split("\n")
                          if line.strip() and line.strip().startswith("-")],
                        ""
                    ])

                # Agent System
                if "Agent System" in phase5_report:
                    agent_section = phase5_report.split("Agent System")[1].split("###")[0]
                    structure_section.extend([
                        "#### Agent System",
                        "**Strengths:**",
                        *[line.strip() for line in agent_section.split("Areas for Improvement")[0].split("\n")
                          if line.strip() and line.strip().startswith("-")],
                        "",
                        "**Areas for Improvement:**",
                        *[line.strip() for line in agent_section.split("Areas for Improvement")[1].split("###")[0].split("\n")
                          if line.strip() and line.strip().startswith("-")],
                        ""
                    ])

                # Documentation Pipeline
                if "Documentation Pipeline" in phase5_report:
                    doc_section = phase5_report.split("Documentation Pipeline")[1].split("##")[0]
                    structure_section.extend([
                        "#### Documentation Pipeline",
                        "**Current Implementation:**",
                        *[line.strip() for line in doc_section.split("Recommended Enhancements")[0].split("\n")
                          if line.strip() and line.strip().startswith("-")],
                        "",
                        "**Recommended Enhancements:**",
                        *[line.strip() for line in doc_section.split("Recommended Enhancements")[1].split("##")[0].split("\n")
                          if line.strip() and line.strip().startswith("-")],
                        ""
                    ])

            # Extract Dependencies
            if "Dependencies" in phase5_report:
                tools_section.extend([
                    "## Dependencies",
                    ""
                ])
                
                # External Requirements
                if "External Requirements" in phase5_report:
                    ext_deps = phase5_report.split("External Requirements")[1].split("```")[1].split("```")[0]
                    tools_section.extend([
                        "### External Requirements",
                        "```python",
                        ext_deps.strip(),
                        "```",
                        ""
                    ])
                
                # Development Dependencies
                if "Development Dependencies" in phase5_report:
                    dev_deps = phase5_report.split("Development Dependencies")[1].split("```")[1].split("```")[0]
                    tools_section.extend([
                        "### Development Dependencies",
                        "```python",
                        dev_deps.strip(),
                        "```",
                        ""
                    ])

        # Add implementation patterns from phase 5
        if "Architectural Patterns" in phase5_report:
            patterns_section = phase5_report.split("Architectural Patterns")[1].split("##")[0]
            infrastructure_section.extend([
                "## Implementation Patterns",
                *[line.strip() for line in patterns_section.split("\n")
                  if line.strip() and line.strip().startswith("-")],
                ""
            ])

        # Combine all sections while preserving template sections
        sections = {}
        current_section = None
        protected_sections = ["Instructions", "Lessons", "Scratchpad"]
        
        for line in template_content.split('\n'):
            if line.startswith('# '):
                current_section = line[2:].strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        # Update non-protected sections
        if "Project Structure" not in protected_sections:
            sections["Project Structure"] = structure_section
        if "Tools" not in protected_sections:
            sections["Tools"] = tools_section
        if "Tech Stack" not in protected_sections:
            sections["Tech Stack"] = tech_stack_section
        if "Infrastructure and DevOps" not in protected_sections:
            sections["Infrastructure and DevOps"] = infrastructure_section

        # Generate final content
        final_content = []
        for section, content in sections.items():
            if section in protected_sections:
                # For protected sections, keep original content including the header
                final_content.append(f"# {section}")
                if isinstance(content, list):
                    final_content.extend(content)
                else:
                    final_content.append(content)
            else:
                # For non-protected sections, only add if there's content
                if content:
                    final_content.append(f"# {section}")
                    if isinstance(content, list):
                        final_content.extend(content)
                    else:
                        final_content.append(content)
            final_content.append("")  # Add empty line between sections

        # Write to .cursorrules file
        cursorrules_path = directory / ".cursorrules"
        with open(cursorrules_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_content))

        logger.info(f"Generated .cursorrules file at: {cursorrules_path}")

    except Exception as e:
        logger.error(f"Error generating .cursorrules file: {str(e)}")
        raise

@click.command()
@click.option('--path', '-p', type=str, help='Path to the project directory')
@click.option('--output', '-o', type=str, help='Output file path')
def main(path: str, output: str):
    """Run multi-phase project analysis"""
    try:
        if not path:
            path = click.prompt('Please provide the project directory path', type=str)
        
        directory = Path(os.path.expanduser(path))
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory path: {path}")
            sys.exit(1)
        
        output_file = output or f"{directory.name}_analysis.txt"
        
        console.print(f"\n[bold]Analyzing project:[/] {directory}")
        analyzer = ProjectAnalyzer(directory)
        start_time = time.time()  # Start timing here
        analysis_result = asyncio.run(analyzer.analyze())
        
        # Save the complete analysis to the main output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        
        # Extract results from analyzer
        phase1_results = analyzer.phase1_results if hasattr(analyzer, 'phase1_results') else {}
        phase2_results = analyzer.phase2_results if hasattr(analyzer, 'phase2_results') else {}
        phase3_results = analyzer.phase3_results if hasattr(analyzer, 'phase3_results') else {}
        phase4_results = analyzer.phase4_results if hasattr(analyzer, 'phase4_results') else {}
        consolidated_report = analyzer.consolidated_report if hasattr(analyzer, 'consolidated_report') else {}
        final_analysis = analyzer.final_analysis if hasattr(analyzer, 'final_analysis') else {}
        
        # Prepare data for individual phase outputs
        analysis_data = {
            "phase1": phase1_results,
            "phase2": phase2_results,
            "phase3": phase3_results,
            "phase4": phase4_results,
            "consolidated_report": consolidated_report,
            "final_analysis": final_analysis,
            "metrics": {
                "time": time.time() - start_time,
                "phase2_tokens": phase2_results.get("reasoning_tokens", 0),
                "phase4_tokens": phase4_results.get("reasoning_tokens", 0),
                "final_tokens": final_analysis.get("reasoning_tokens", 0)
            }
        }
        
        # Save individual phase outputs
        save_phase_outputs(directory, analysis_data)
        
        console.print(f"\n[green]Analysis saved to:[/] {output_file}")
        console.print(f"[green]Individual phase outputs saved to:[/] {directory}/phases_output/")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
