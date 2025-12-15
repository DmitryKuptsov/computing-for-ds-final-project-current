from src.home_credit.pipeline import PipelineRunner


def main() -> None:
    """Run the full Home Credit regression pipeline."""
    runner = PipelineRunner("config/default.yaml")
    runner.run()


if __name__ == "__main__":
    main()
