from pathlib import Path

from src.langgraph_agent import agent


def save_graph_image(output_path: str = "docs/graph.png") -> None:
    """Render and save the LangGraph diagram as PNG.

    Falls back to printing Mermaid syntax when PNG rendering dependencies
    are not available.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        graph_png = agent.get_graph().draw_mermaid_png()
        output.write_bytes(graph_png)
        print(f"Graph saved to: {output}")
    except Exception:
        print("Could not generate PNG graph. Ensure rendering dependencies are installed.")
        print("Use this Mermaid in mermaid.live:")
        print(agent.get_graph().draw_mermaid())


if __name__ == "__main__":
    save_graph_image()
