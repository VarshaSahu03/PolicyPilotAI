
const API_URL = "http://127.0.0.1:8000";

export async function sendMessageToAPI(query, onToken) {

  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    let errorMessage = `Backend error: ${response.status}`;

    try {
      const data = await response.json();
      errorMessage = data.detail || errorMessage;
    } catch {}

    throw new Error(errorMessage);
  }

  if (!response.body) {
    throw new Error("Streaming is not supported by this response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  try {
    while (true) {

      const { value, done } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, {
        stream: true
      });

      const events = buffer.split("\n\n");

      buffer = events.pop() || "";

      for (const event of events) {

        const dataLine = event
          .split("\n")
          .find(line => line.startsWith("data: "));

        if (!dataLine) continue;

        const data = JSON.parse(dataLine.slice(6));

        if (data.error) {
          throw new Error(data.error);
        }

        if (data.token) {
          onToken(data.token);
        }

        if (data.done) {
          return;
        }
      }
    }

    buffer += decoder.decode();

  } finally {
    reader.releaseLock();
  }
}
