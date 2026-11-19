
import { useState, useRef, useEffect } from "react";
import MarkdownAnswer from "./MarkdownAnswer";
import "./App.css";
import { sendMessageToAPI } from "./api";

const STARTER_PROMPTS = [
  "Tell me about Maternity leave policy?",
  "How to hack password?",
  "What is Voluntary Retirement Scheme (VRS)?",
  "Explain the notice period and exit guidelines."
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [openSources, setOpenSources] = useState({});

  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    });
  }, [messages, loading]);

  // Send message to FastAPI
  const handleSendMessage = async (textToSend) => {
    const query = (textToSend ?? input).trim();

    if (!query || loading) return;

    setInput("");
    setLoading(true);

    // Add user and assistant messages
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: query
      },
      {
        role: "assistant",
        content: "",
        sources: []
      }
    ]);

    try {
      // Receive streaming tokens
      await sendMessageToAPI(query, (token) => {

        if (typeof token !== "string") return;

        setMessages((prev) => {
          const updated = [...prev];

          const lastIndex = updated.length - 1;

          if (
            lastIndex >= 0 &&
            updated[lastIndex].role === "assistant"
          ) {
            updated[lastIndex] = {
              ...updated[lastIndex],
              content: updated[lastIndex].content + token
            };
          }

          return updated;
        });

      });

    } catch (error) {

      // Handle API errors
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;

        if (
          lastIndex >= 0 &&
          updated[lastIndex].role === "assistant"
        ) {
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: `⚠️ ${error.message || "Something went wrong. Please try again."}`
          };
        }

        return updated;
      });

    } finally {
      setLoading(false);
    }
  };

  // Enter to send, Shift + Enter for new line
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Toggle policy sources
  const toggleSources = (idx) => {
    setOpenSources((prev) => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  // Start new chat
  const startNewChat = () => {
    if (loading) return;

    setMessages([]);
    setOpenSources({});
    setInput("");
  };

  return (
    <div className="layout">

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>

        <div className="sidebar-header">
          <button
            className="new-chat-btn"
            onClick={startNewChat}
            disabled={loading}
          >
            <span>+</span> New chat
          </button>
        </div>

        <div className="sidebar-info">
          <div className="badge">
            Internal Knowledge Base
          </div>

          <p>
            Connected to indexed company policies via
            Hybrid RAG (BM25 + ChromaDB).
          </p>
        </div>

        <div className="sidebar-footer">
          <div className="brand-dot"></div>
          <span>PolicyPilotAI</span>
        </div>

      </aside>

      {/* Main Chat */}
      <main className="chat-container">

        {/* Top Bar */}
        <header className="top-bar">

          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((prev) => !prev)}
            title="Toggle Sidebar"
          >
            ☰
          </button>

          <span className="model-title">
            PolicyPilotAI
            <span className="version"> v1.0</span>
          </span>

        </header>

        {/* Message Area */}
        <div className="chat-scroll-area">

          {messages.length === 0 ? (

            /* Welcome Screen */
            <div className="welcome-screen">

              <div className="logo-circle">P</div>

              <h1>
                How can I help with company policies today?
              </h1>

              <div className="starter-grid">

                {STARTER_PROMPTS.map((prompt, index) => (
                  <button
                    key={index}
                    className="starter-card"
                    onClick={() => handleSendMessage(prompt)}
                    disabled={loading}
                  >
                    {prompt}
                  </button>
                ))}

              </div>

            </div>

          ) : (

            /* Chat Messages */
            <div className="messages-wrapper">

              {messages.map((msg, idx) => (

                <div
                  key={idx}
                  className={`message-row ${msg.role}`}
                >

                  <div className="message-content-box">

                    {/* Message Content */}
                    <div className="message-text">
                      <MarkdownAnswer content = {msg.content} />
                    </div>

                    {/* Policy Sources */}
                    {msg.sources?.length > 0 && (

                      <div className="sources-container">

                        <button
                          className="sources-toggle"
                          onClick={() => toggleSources(idx)}
                        >
                          📄 {openSources[idx] ? "Hide" : "View"}{" "}
                          {msg.sources.length} policy sources
                        </button>

                        {openSources[idx] && (

                          <div className="sources-list">

                            {msg.sources.map((src, sIdx) => (

                              <div
                                key={sIdx}
                                className="source-item"
                              >

                                <div className="source-header">
                                  <strong>{src.file}</strong>
                                  {" "}• Page {src.page}
                                </div>

                                {src.snippet && (
                                  <p className="source-snippet">
                                    "{src.snippet}"
                                  </p>
                                )}

                              </div>

                            ))}

                          </div>

                        )}

                      </div>

                    )}

                  </div>

                </div>

              ))}

              {/* Typing Indicator */}
              {loading &&
                messages[messages.length - 1]?.content === "" && (

                  <div className="message-row assistant">

                    <div className="avatar assistant-avatar">
                      P
                    </div>

                    <div className="message-content-box">

                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>

                    </div>

                  </div>

                )}

              {/* Auto Scroll Reference */}
              <div ref={messagesEndRef} />

            </div>

          )}

        </div>

        {/* Input Area */}
        <div className="input-area-wrapper">

          <div className="input-box">

            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message PolicyPilotAI..."
              disabled={loading}
            />

            <button
              className="send-btn"
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || loading}
              title="Send message"
            >
              ↑
            </button>

          </div>

          <p className="disclaimer">
            Answers are retrieved strictly from internal
            company policy documents.
          </p>

        </div>

      </main>

    </div>
  );
}

export default App;
