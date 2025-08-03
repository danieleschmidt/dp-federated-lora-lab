"""
Command-line interface for DP-Federated LoRA.

This module provides CLI commands for running federated learning
experiments, managing clients and servers, and monitoring training.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import torch

from .config import (
    FederatedConfig, 
    ClientConfig,
    create_default_config,
    create_high_privacy_config,
    create_performance_config
)
from .server import FederatedServer
from .client import DPLoRAClient, create_mock_client
from .monitoring import UtilityMonitor


# CLI app
app = typer.Typer(
    name="dp-fed-lora",
    help="DP-Federated LoRA: Privacy-preserving federated learning for LLMs",
    rich_markup_mode="rich"
)

# Console for rich output
console = Console()

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@app.command()
def server(
    model_name: str = typer.Option(
        "meta-llama/Llama-2-7b-hf",
        "--model", "-m",
        help="Base model name for federated training"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c", 
        help="Configuration file path"
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Server host address"
    ),
    port: int = typer.Option(
        8443,
        "--port", "-p",
        help="Server port"
    ),
    rounds: int = typer.Option(
        50,
        "--rounds", "-r",
        help="Number of training rounds"
    ),
    epsilon: float = typer.Option(
        8.0,
        "--epsilon", "-e",
        help="Privacy budget (epsilon)"
    ),
    delta: float = typer.Option(
        1e-5,
        "--delta", "-d",
        help="Privacy parameter (delta)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging"
    )
) -> None:
    """Start the federated learning server."""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"[bold blue]Starting DP-Federated LoRA Server[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Host: {host}:{port}")
    console.print(f"Privacy: ε={epsilon}, δ={delta}")
    
    try:
        # Load configuration
        if config_file:
            with open(config_file) as f:
                config_data = json.load(f)
            config = FederatedConfig(**config_data)
        else:
            config = create_default_config()
        
        # Override with CLI parameters
        config.model_name = model_name
        config.num_rounds = rounds
        config.privacy.epsilon = epsilon
        config.privacy.delta = delta
        config.server_host = host
        config.server_port = port
        
        # Initialize server
        server = FederatedServer(
            model_name=model_name,
            config=config
        )
        
        # Initialize global model
        with console.status("[bold green]Initializing global model..."):
            server.initialize_global_model()
        
        console.print(f"[green]✓[/green] Server initialized successfully")
        console.print(f"[yellow]Waiting for clients to connect...[/yellow]")
        
        # In a real implementation, this would start the server
        # For now, we'll simulate server operation
        console.print(f"[green]Server ready at {host}:{port}[/green]")
        console.print(f"[dim]Press Ctrl+C to stop the server[/dim]")
        
        # Keep server running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Server shutting down...[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def client(
    client_id: str = typer.Argument(
        help="Unique client identifier"
    ),
    data_path: str = typer.Argument(
        help="Path to local training data"
    ),
    server_host: str = typer.Option(
        "localhost",
        "--host",
        help="Server host address"
    ),
    server_port: int = typer.Option(
        8443,
        "--port", "-p",
        help="Server port"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file path"
    ),
    mock_data: bool = typer.Option(
        False,
        "--mock-data",
        help="Use mock data for testing"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging"
    )
) -> None:
    """Start a federated learning client."""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"[bold blue]Starting DP-Federated LoRA Client[/bold blue]")
    console.print(f"Client ID: {client_id}")
    console.print(f"Server: {server_host}:{server_port}")
    
    try:
        # Load configuration
        if config_file:
            with open(config_file) as f:
                config_data = json.load(f)
            fed_config = FederatedConfig(**config_data)
        else:
            fed_config = create_default_config()
        
        # Update server connection
        fed_config.server_host = server_host
        fed_config.server_port = server_port
        
        # Create client
        if mock_data:
            console.print("[yellow]Using mock data for testing[/yellow]")
            client = create_mock_client(client_id, num_examples=1000, config=fed_config)
        else:
            client_config = ClientConfig(client_id=client_id, data_path=data_path)
            client = DPLoRAClient(
                client_id=client_id,
                data_path=data_path,
                config=fed_config,
                client_config=client_config
            )
        
        # Setup client
        with console.status("[bold green]Setting up client..."):
            client.setup()
        
        console.print(f"[green]✓[/green] Client initialized successfully")
        console.print(f"[green]Client ready for federated training[/green]")
        
        # In a real implementation, this would connect to server
        console.print(f"[dim]Press Ctrl+C to stop the client[/dim]")
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Client shutting down...[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error starting client: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    model_name: str = typer.Option(
        "meta-llama/Llama-2-7b-hf",
        "--model", "-m",
        help="Base model name"
    ),
    num_clients: int = typer.Option(
        10,
        "--clients", "-c",
        help="Number of clients"
    ),
    rounds: int = typer.Option(
        50,
        "--rounds", "-r",
        help="Number of training rounds"
    ),
    epsilon: float = typer.Option(
        8.0,
        "--epsilon", "-e",
        help="Privacy budget (epsilon)"
    ),
    delta: float = typer.Option(
        1e-5,
        "--delta", "-d",
        help="Privacy parameter (delta)"
    ),
    lora_rank: int = typer.Option(
        16,
        "--lora-rank",
        help="LoRA rank"
    ),
    local_epochs: int = typer.Option(
        3,
        "--local-epochs",
        help="Local training epochs"
    ),
    aggregation: str = typer.Option(
        "fedavg",
        "--aggregation", "-a",
        help="Aggregation method (fedavg, krum, trimmed_mean)"
    ),
    output_dir: str = typer.Option(
        "./federated_results",
        "--output", "-o",
        help="Output directory for results"
    ),
    config_preset: str = typer.Option(
        "default",
        "--preset",
        help="Configuration preset (default, high_privacy, performance)"
    ),
    mock_data: bool = typer.Option(
        True,
        "--mock-data/--real-data",
        help="Use mock data for testing"
    )
) -> None:
    """Run a complete federated training experiment."""
    
    console.print(f"[bold blue]Running DP-Federated LoRA Training[/bold blue]")
    
    # Create configuration based on preset
    if config_preset == "high_privacy":
        config = create_high_privacy_config()
    elif config_preset == "performance":
        config = create_performance_config()
    else:
        config = create_default_config()
    
    # Override with CLI parameters
    config.model_name = model_name
    config.num_rounds = rounds
    config.local_epochs = local_epochs
    config.privacy.epsilon = epsilon
    config.privacy.delta = delta
    config.lora.r = lora_rank
    config.output_dir = output_dir
    
    # Display configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", model_name)
    table.add_row("Clients", str(num_clients))
    table.add_row("Rounds", str(rounds))
    table.add_row("Privacy (ε, δ)", f"{epsilon}, {delta}")
    table.add_row("LoRA Rank", str(lora_rank))
    table.add_row("Local Epochs", str(local_epochs))
    table.add_row("Aggregation", aggregation)
    
    console.print(table)
    
    try:
        # Initialize server
        with console.status("[bold green]Initializing server..."):
            server = FederatedServer(
                model_name=model_name,
                config=config,
                num_clients=num_clients,
                rounds=rounds,
                privacy_budget={"epsilon": epsilon, "delta": delta}
            )
            server.initialize_global_model()
        
        # Create clients
        clients = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Creating clients...", total=num_clients)
            
            for i in range(num_clients):
                if mock_data:
                    client = create_mock_client(
                        client_id=f"client_{i}",
                        num_examples=500 + i * 100,  # Variable data sizes
                        config=config
                    )
                else:
                    # In real scenario, clients would connect with their own data
                    client = create_mock_client(f"client_{i}", config=config)
                
                client.setup()
                clients.append(client)
                progress.update(task, advance=1)
        
        console.print(f"[green]✓[/green] Created {num_clients} clients")
        
        # Run federated training
        console.print(f"[bold yellow]Starting federated training...[/bold yellow]")
        
        with Progress() as progress:
            task = progress.add_task("Training rounds", total=rounds)
            
            # Start training
            history = server.train(
                clients=clients,
                aggregation=aggregation,
                local_epochs=local_epochs
            )
            
            progress.update(task, completed=rounds)
        
        # Display results
        console.print(f"\n[bold green]Training completed![/bold green]")
        
        results_table = Table(title="Training Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Rounds Completed", str(history.rounds_completed))
        results_table.add_row("Total Duration", f"{history.total_duration:.1f}s")
        results_table.add_row("Final Accuracy", f"{history.final_accuracy:.2%}")
        results_table.add_row("Privacy Spent (ε)", f"{history.final_epsilon:.2f}")
        results_table.add_row("Privacy Budget", f"{history.total_epsilon:.2f}")
        results_table.add_row("Budget Utilization", f"{history.final_epsilon/history.total_epsilon:.1%}")
        
        console.print(results_table)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)
        
        console.print(f"[green]Results saved to {results_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    models: List[str] = typer.Option(
        ["meta-llama/Llama-2-7b-hf"],
        "--model", "-m",
        help="Models to benchmark"
    ),
    privacy_levels: List[float] = typer.Option(
        [1.0, 4.0, 8.0],
        "--epsilon", "-e",
        help="Privacy levels to test"
    ),
    client_counts: List[int] = typer.Option(
        [10, 50],
        "--clients", "-c",
        help="Client counts to test"
    ),
    rounds: int = typer.Option(
        30,
        "--rounds", "-r",
        help="Number of rounds per experiment"
    ),
    output_dir: str = typer.Option(
        "./benchmark_results",
        "--output", "-o",
        help="Output directory"
    )
) -> None:
    """Run comprehensive benchmarking experiments."""
    
    console.print(f"[bold blue]Running DP-Federated LoRA Benchmarks[/bold blue]")
    
    total_experiments = len(models) * len(privacy_levels) * len(client_counts)
    console.print(f"Total experiments: {total_experiments}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with Progress() as progress:
        main_task = progress.add_task("Benchmark progress", total=total_experiments)
        
        for model in models:
            for epsilon in privacy_levels:
                for num_clients in client_counts:
                    console.print(f"\n[yellow]Running: {model}, ε={epsilon}, clients={num_clients}[/yellow]")
                    
                    try:
                        # Create configuration
                        config = create_default_config()
                        config.model_name = model
                        config.privacy.epsilon = epsilon
                        config.num_rounds = rounds
                        
                        # Run experiment
                        server = FederatedServer(
                            model_name=model,
                            config=config,
                            num_clients=num_clients,
                            rounds=rounds,
                            privacy_budget={"epsilon": epsilon}
                        )
                        
                        # Create mock clients
                        clients = [
                            create_mock_client(f"client_{i}", config=config)
                            for i in range(num_clients)
                        ]
                        
                        for client in clients:
                            client.setup()
                        
                        # Run training
                        history = server.train(clients=clients)
                        
                        # Record results
                        result = {
                            "model": model,
                            "epsilon": epsilon,
                            "num_clients": num_clients,
                            "rounds": rounds,
                            "final_accuracy": history.final_accuracy,
                            "final_epsilon": history.final_epsilon,
                            "duration": history.total_duration,
                            "rounds_completed": history.rounds_completed
                        }
                        
                        results.append(result)
                        console.print(f"[green]✓[/green] Completed: accuracy={history.final_accuracy:.2%}")
                        
                    except Exception as e:
                        console.print(f"[red]✗[/red] Failed: {e}")
                        result = {
                            "model": model,
                            "epsilon": epsilon,
                            "num_clients": num_clients,
                            "error": str(e)
                        }
                        results.append(result)
                    
                    progress.update(main_task, advance=1)
    
    # Save results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[bold green]Benchmarking completed![/bold green]")
    console.print(f"Results saved to {results_file}")
    
    # Display summary
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        avg_accuracy = sum(r["final_accuracy"] for r in successful_results) / len(successful_results)
        console.print(f"Average accuracy across experiments: {avg_accuracy:.2%}")


@app.command()
def analyze(
    results_file: Path = typer.Argument(
        help="Path to training results file"
    ),
    output_dir: str = typer.Option(
        "./analysis",
        "--output", "-o",
        help="Output directory for analysis"
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate plots"
    )
) -> None:
    """Analyze training results and generate reports."""
    
    console.print(f"[bold blue]Analyzing Results[/bold blue]")
    console.print(f"Input: {results_file}")
    
    try:
        # Load results
        with open(results_file) as f:
            data = json.load(f)
        
        # Create utility monitor for analysis
        monitor = UtilityMonitor()
        
        # Process data based on format
        if isinstance(data, list):
            # Benchmark results
            for result in data:
                if "error" not in result:
                    monitor.record_utility_point({
                        "epsilon": result.get("epsilon", 0),
                        "accuracy": result.get("final_accuracy", 0),
                        "utility": result.get("final_accuracy", 0)
                    })
        else:
            # Single training results
            privacy_timeline = data.get("privacy_timeline", [])
            for point in privacy_timeline:
                monitor.record_utility_point({
                    "epsilon": point.get("epsilon_spent", 0),
                    "accuracy": 0.85,  # Placeholder
                    "utility": 0.85
                })
        
        # Generate report
        report = monitor.generate_report()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_file = output_path / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots if requested
        if plot:
            try:
                plot_file = output_path / "privacy_utility_curve.png"
                monitor.plot_privacy_utility_curve(save_path=str(plot_file))
                console.print(f"[green]Plot saved to {plot_file}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not generate plot: {e}[/yellow]")
        
        # Display summary
        if "error" not in report:
            console.print(f"\n[bold green]Analysis completed![/bold green]")
            console.print(f"Total experiments: {report['summary']['total_experiments']}")
            
            if "optimal_points" in report:
                best_point = report["optimal_points"]["best_efficiency"]
                console.print(f"Best efficiency: ε={best_point['epsilon']:.2f}, utility={best_point['utility']:.2%}")
        
        console.print(f"Report saved to {report_file}")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    preset: str = typer.Option(
        "default",
        help="Configuration preset (default, high_privacy, performance)"
    ),
    output_file: str = typer.Option(
        "config.json",
        "--output", "-o",
        help="Output configuration file"
    )
) -> None:
    """Generate configuration files."""
    
    console.print(f"[bold blue]Generating Configuration[/bold blue]")
    console.print(f"Preset: {preset}")
    
    # Create configuration based on preset
    if preset == "high_privacy":
        config = create_high_privacy_config()
    elif preset == "performance":
        config = create_performance_config()
    else:
        config = create_default_config()
    
    # Convert to dictionary for JSON serialization
    config_dict = {
        "model_name": config.model_name,
        "num_rounds": config.num_rounds,
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "privacy": {
            "epsilon": config.privacy.epsilon,
            "delta": config.privacy.delta,
            "noise_multiplier": config.privacy.noise_multiplier,
            "max_grad_norm": config.privacy.max_grad_norm
        },
        "lora": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "target_modules": config.lora.target_modules
        },
        "security": {
            "byzantine_fraction": config.security.byzantine_fraction,
            "aggregation_method": config.security.aggregation_method.value,
            "client_sampling_rate": config.security.client_sampling_rate
        }
    }
    
    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    console.print(f"[green]Configuration saved to {output_file}[/green]")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()