import React, { Component, KeyboardEvent, createRef } from "react";
import { isRecord } from "./record";
import "./style.css";

type Sender = "assistant" | "user";

type Message = {
  id: string;
  sender: Sender;
  content: string;
  timestamp: string;
};

type ChatAppState = {
  messages: Message[];
  inputText: string;
  isLoading: boolean;
  errorMsg: string | undefined;
  siteUrl: string;
  embedMode: boolean;
};

const nowTimestamp = (): string => {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
};

const readInitialSiteUrl = (): string => {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("siteUrl");
  if (typeof fromQuery === "string" && fromQuery.trim().length > 0) {
    return fromQuery;
  }
  return window.location.origin;
};

const readEmbedMode = (): boolean => {
  const params = new URLSearchParams(window.location.search);
  return params.get("embed") === "1";
};

/** Displays the UI of the chat application. */
export class ChatApp extends Component<{}, ChatAppState> {
  private readonly messagesRef = createRef<HTMLDivElement>();

  constructor(props: {}) {
    super(props);

    this.state = {
      messages: [
        {
          id: "assistant-welcome",
          sender: "assistant",
          content: "Ask me anything about this website.",
          timestamp: nowTimestamp(),
        },
      ],
      inputText: "",
      isLoading: false,
      errorMsg: undefined,
      siteUrl: readInitialSiteUrl(),
      embedMode: readEmbedMode(),
    };
  }

  componentDidUpdate = (prevProps: {}, prevState: ChatAppState): void => {
    if (
      prevState.messages.length !== this.state.messages.length ||
      prevState.isLoading !== this.state.isLoading ||
      prevState.errorMsg !== this.state.errorMsg
    ) {
      this.scrollToBottom();
    }
  };

  render = (): JSX.Element => {
    const shellClass = this.state.embedMode
      ? "WidgetRoot WidgetRoot--embedded"
      : "WidgetRoot";

    const displayDomain = this.getDisplayDomain(this.state.siteUrl);

    return (
      <main className={shellClass}>
        <section className="ChatShell" aria-label="Website assistant">
          <header className="ChatHeader">
            <div className="ChatHeader-title">Website Assistant</div>
            <div className="ChatHeader-subtitle">
              Answers from {displayDomain}
            </div>
          </header>

          <div className="ChatWindow" ref={this.messagesRef}>
            {this.renderMessages()}
            {this.renderTypingState()}
            {this.renderErrorState()}
          </div>

          <form className="ChatInput" onSubmit={this.onSubmit}>
            <textarea
              className="ChatInput-input"
              value={this.state.inputText}
              placeholder="Ask about this website..."
              onChange={this.onInputChange}
              onKeyDown={this.onInputKeyDown}
              rows={2}
              disabled={this.state.isLoading}
            />
            <button
              className="ChatInput-btnSend"
              type="submit"
              disabled={this.state.isLoading || this.state.inputText.trim().length === 0}
            >
              Send
            </button>
          </form>
        </section>
      </main>
    );
  };

  renderMessages = (): JSX.Element => {
    return (
      <div>
        {this.state.messages.map((message) => {
          const itemClass =
            message.sender === "assistant"
              ? "ChatItem ChatItem--assistant"
              : "ChatItem ChatItem--user";
          const label = message.sender === "assistant" ? "Assistant" : "You";

          return (
            <div className={itemClass} key={message.id}>
              <div className="ChatItem-chatContent">
                <div className="ChatItem-chatText">{message.content}</div>
                <div className="ChatItem-timeStamp">
                  <strong>{label}</strong> · {message.timestamp}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  renderTypingState = (): JSX.Element | null => {
    if (!this.state.isLoading) {
      return null;
    }

    return (
      <div className="ChatItem ChatItem--assistant">
        <div className="ChatItem-chatContent">
          <div className="ChatItem-chatText ChatItem-chatText--typing">
            Thinking...
          </div>
        </div>
      </div>
    );
  };

  renderErrorState = (): JSX.Element | null => {
    if (typeof this.state.errorMsg === "undefined") {
      return null;
    }

    return <div className="ChatError">{this.state.errorMsg}</div>;
  };

  onInputChange = (evt: React.ChangeEvent<HTMLTextAreaElement>): void => {
    this.setState({ inputText: evt.currentTarget.value });
  };

  onInputKeyDown = (evt: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      this.submitPrompt();
    }
  };

  onSubmit = (evt: React.FormEvent<HTMLFormElement>): void => {
    evt.preventDefault();
    this.submitPrompt();
  };

  submitPrompt = (): void => {
    const query = this.state.inputText.trim();
    if (query.length === 0 || this.state.isLoading) {
      return;
    }

    const message: Message = {
      id: `user-${Date.now()}`,
      sender: "user",
      content: query,
      timestamp: nowTimestamp(),
    };

    this.setState(
      (prevState) => ({
        messages: prevState.messages.concat([message]),
        inputText: "",
        isLoading: true,
        errorMsg: undefined,
      }),
      () => {
        this.doQuerySubmitted(query);
      }
    );
  };

  doQuerySubmitted = async (query: string): Promise<void> => {
    const params = new URLSearchParams({
      prompt: query,
      siteUrl: this.state.siteUrl,
      pageUrl: window.location.href,
    });

    try {
      const res = await fetch(`/api/respond?${params.toString()}`);
      if (res.status !== 200) {
        this.doQueryError(`Bad status code ${res.status}`);
        return;
      }

      const payload: unknown = await res.json();
      this.doQueryJson(payload);
    } catch (_err) {
      this.doQueryError("Failed to connect to server");
    }
  };

  doQueryJson = (data: unknown): void => {
    if (!isRecord(data)) {
      this.doQueryError("Invalid server payload");
      return;
    }

    if (typeof data.response !== "string") {
      this.doQueryError("Expected response to be a string");
      return;
    }

    const message: Message = {
      id: `assistant-${Date.now()}`,
      sender: "assistant",
      content: data.response,
      timestamp: nowTimestamp(),
    };

    this.setState((prevState) => ({
      messages: prevState.messages.concat([message]),
      isLoading: false,
      errorMsg: undefined,
    }));
  };

  doQueryError = (msg: string): void => {
    this.setState({
      isLoading: false,
      errorMsg: msg,
    });
    console.error(`Error fetching /api/respond: ${msg}`);
  };

  scrollToBottom = (): void => {
    if (this.messagesRef.current !== null) {
      this.messagesRef.current.scrollTop = this.messagesRef.current.scrollHeight;
    }
  };

  getDisplayDomain = (value: string): string => {
    try {
      const url = new URL(value);
      return url.hostname;
    } catch (_err) {
      return value;
    }
  };
}
