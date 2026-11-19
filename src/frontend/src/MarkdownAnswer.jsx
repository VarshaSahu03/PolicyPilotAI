
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./MarkdownAnswer.css";

export default function MarkdownAnswer({ content }) {
  return (
    <div className="ai-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ ...props }) => (
            <p className="answer-paragraph" {...props} />
          ),

          strong: ({ ...props }) => (
            <strong className="answer-bold" {...props} />
          ),

          ul: ({ ...props }) => (
            <ul className="answer-list" {...props} />
          ),

          ol: ({ ...props }) => (
            <ol className="answer-ordered-list" {...props} />
          ),

          li: ({ ...props }) => (
            <li className="answer-list-item" {...props} />
          ),

          h1: ({ ...props }) => (
            <h1 className="answer-heading" {...props} />
          ),

          h2: ({ ...props }) => (
            <h2 className="answer-heading" {...props} />
          ),

          h3: ({ ...props }) => (
            <h3 className="answer-subheading" {...props} />
          ),

          table: ({ ...props }) => (
            <div className="answer-table-wrapper">
              <table className="answer-table" {...props} />
            </div>
          ),

          blockquote: ({ ...props }) => (
            <blockquote className="answer-quote" {...props} />
          ),

          a: ({ ...props }) => (
            <a
              className="answer-link"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),

          pre: ({ ...props }) => (
            <pre className="answer-code-block" {...props} />
          ),

          hr: () => <hr className="answer-divider" />,
        }}
      >
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}
