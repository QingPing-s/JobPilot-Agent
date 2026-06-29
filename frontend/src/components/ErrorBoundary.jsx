import React from "react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="error-boundary">
          <h1>页面加载失败</h1>
          <p>{this.state.error.message || "前端发生未处理错误。"}</p>
          <button onClick={() => window.location.reload()} type="button">
            重新加载
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}
