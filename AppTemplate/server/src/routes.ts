//(C) Shreyan Mitra
import { Request, Response } from "express";
import { ParamsDictionary } from "express-serve-static-core";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { compile } from "html-to-text";

require("dotenv").config();

type SafeRequest = Request<ParamsDictionary, {}, Record<string, unknown>>;
type SafeResponse = Response;

type ChatbotSession = {
  chatbot: any;
  contextGenerator: any;
};

type LlmProvider = "openai" | "ollama";

const sessions = new Map<string, ChatbotSession>();
const pendingSessions = new Map<string, Promise<ChatbotSession>>();

const getProvider = (): LlmProvider => {
  const raw = (process.env.LLM_PROVIDER ?? "ollama").toLowerCase();
  if (raw === "openai") {
    return "openai";
  }
  return "ollama";
};

const getRequiredEnv = (name: string): string => {
  const value = process.env[name];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
};

const getOllamaBaseUrl = (): string => {
  const explicit = process.env.OLLAMA_BASE_URL?.trim();
  if (typeof explicit === "string" && explicit !== "") {
    return explicit;
  }

  const openWebUI = process.env.OPEN_WEBUI_URL?.trim();
  if (typeof openWebUI === "string" && openWebUI !== "") {
    return `${openWebUI.replace(/\/$/, "")}/ollama`;
  }

  return "http://127.0.0.1:11434";
};

const createChatModel = (): ChatOpenAI | ChatOllama => {
  const provider = getProvider();
  if (provider === "openai") {
    return new ChatOpenAI({
      model: process.env.OPENAI_MODEL ?? "gpt-4o",
      apiKey: getRequiredEnv("OPENAI_API_KEY"),
      temperature: 0.6,
    });
  }

  return new ChatOllama({
    model: process.env.OLLAMA_MODEL ?? "llama3.1:8b",
    baseUrl: getOllamaBaseUrl(),
    temperature: 0.4,
  });
};

const createEmbeddingsModel = (): OpenAIEmbeddings | OllamaEmbeddings => {
  const provider = getProvider();
  if (provider === "openai") {
    return new OpenAIEmbeddings({
      model: process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small",
      apiKey: getRequiredEnv("OPENAI_API_KEY"),
    });
  }

  return new OllamaEmbeddings({
    model: process.env.OLLAMA_EMBEDDING_MODEL ?? "nomic-embed-text",
    baseUrl: getOllamaBaseUrl(),
  });
};

const createSession = async (siteUrl: string): Promise<ChatbotSession> => {
  const llm = createChatModel();

  const compiledConvert = compile({ wordwrap: 130 });
  const loader = new RecursiveUrlLoader(siteUrl, { extractor: compiledConvert });
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splits,
    createEmbeddingsModel()
  );

  const contextGenerator = vectorStore.asRetriever();
  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
  const chatbot = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  return {
    chatbot,
    contextGenerator,
  };
};

const getOrCreateSession = async (siteUrl: string): Promise<ChatbotSession> => {
  const existing = sessions.get(siteUrl);
  if (typeof existing !== "undefined") {
    return existing;
  }

  const pending = pendingSessions.get(siteUrl);
  if (typeof pending !== "undefined") {
    return pending;
  }

  const sessionPromise = createSession(siteUrl)
    .then((session) => {
      sessions.set(siteUrl, session);
      pendingSessions.delete(siteUrl);
      return session;
    })
    .catch((err: unknown) => {
      pendingSessions.delete(siteUrl);
      throw err;
    });

  pendingSessions.set(siteUrl, sessionPromise);
  return sessionPromise;
};

const getChatbotAnswer = async (
  session: ChatbotSession,
  prompt: string,
  pageUrl: string | undefined
): Promise<string> => {
  const userPrompt =
    typeof pageUrl === "string" && pageUrl.length > 0
      ? `Current page: ${pageUrl}\n\nQuestion: ${prompt}`
      : prompt;

  const retrievedDocs = await session.contextGenerator.invoke(userPrompt);
  const response = await session.chatbot.invoke({
    question: userPrompt,
    context: retrievedDocs,
  });

  return response;
};

const normalizeUrl = (raw: string): string | undefined => {
  const cleaned = raw.trim();
  if (cleaned.length === 0) {
    return undefined;
  }

  try {
    const url = new URL(cleaned);
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch (_err) {
    try {
      const url = new URL(`https://${cleaned}`);
      url.search = "";
      url.hash = "";
      return url.toString();
    } catch (_fallbackErr) {
      return undefined;
    }
  }
};

const readSiteUrl = (req: SafeRequest): string | undefined => {
  const fromQuery = first(req.query.siteUrl);
  if (typeof fromQuery === "string") {
    return normalizeUrl(fromQuery);
  }

  const fromEnv = process.env.URL;
  if (typeof fromEnv === "string") {
    return normalizeUrl(fromEnv);
  }

  return undefined;
};

/**
 * Get the chatbot's reponse to a prompt
 * @param req A container for the client's request
 * @param res A container for the server's response
 */
export const respond = async (req: SafeRequest, res: SafeResponse): Promise<void> => {
  const userEntry: string | undefined = first(req.query.prompt);
  if (userEntry === undefined || userEntry.trim() === "") {
    res.status(400).send("Malformed prompt.");
    return;
  }

  const siteUrl = readSiteUrl(req);
  if (typeof siteUrl === "undefined") {
    res.status(400).send("Missing or invalid siteUrl.");
    return;
  }

  const pageUrl = normalizeUrl(first(req.query.pageUrl) ?? "");

  try {
    const session = await getOrCreateSession(siteUrl);
    const response = await getChatbotAnswer(session, userEntry, pageUrl);
    res.send({ response });
  } catch (_err) {
    res.status(500).send("Something went wrong on our end :(");
  }
};

/**
 * Serves a drop-in script that injects an iframe widget on any website.
 */
export const embedScript = (_req: SafeRequest, res: SafeResponse): void => {
  res.setHeader("Content-Type", "application/javascript; charset=utf-8");
  res.send(`(function () {
  var script = document.currentScript;
  if (!script) return;

  var scriptUrl = new URL(script.src, window.location.href);
  var assistantBase = script.dataset.assistantBase || scriptUrl.origin;
  var siteUrl = script.dataset.siteUrl || window.location.origin;
  var title = script.dataset.title || "Ask AI";
  var zIndex = script.dataset.zIndex || "2147483000";

  var launcher = document.createElement("button");
  launcher.type = "button";
  launcher.setAttribute("aria-label", title);
  launcher.innerText = title;

  var panel = document.createElement("div");
  panel.setAttribute("aria-hidden", "true");

  var iframe = document.createElement("iframe");
  iframe.title = "Website Assistant";
  iframe.loading = "lazy";
  iframe.src = assistantBase + "/?embed=1&siteUrl=" + encodeURIComponent(siteUrl);
  iframe.style.width = "100%";
  iframe.style.height = "100%";
  iframe.style.border = "0";
  panel.appendChild(iframe);

  var style = document.createElement("style");
  style.textContent = ""
    + ".wa-launcher{position:fixed;right:20px;bottom:20px;z-index:" + zIndex + ";"
    + "border:0;border-radius:999px;padding:13px 18px;font:600 14px/1 sans-serif;"
    + "background:#e36d3f;color:#fff;cursor:pointer;box-shadow:0 10px 24px rgba(0,0,0,.2);}"
    + ".wa-panel{position:fixed;right:20px;bottom:76px;width:min(420px,calc(100vw - 20px));"
    + "height:min(640px,calc(100vh - 100px));z-index:" + zIndex + ";display:none;"
    + "background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 18px 36px rgba(0,0,0,.26);}"
    + ".wa-panel.is-open{display:block;}"
    + "@media (max-width:640px){.wa-panel{right:0;bottom:0;width:100vw;height:100dvh;border-radius:0;}"
    + ".wa-launcher{right:12px;bottom:12px;}}";

  launcher.className = "wa-launcher";
  panel.className = "wa-panel";

  launcher.addEventListener("click", function () {
    var open = panel.classList.toggle("is-open");
    panel.setAttribute("aria-hidden", open ? "false" : "true");
    launcher.innerText = open ? "Close" : title;
  });

  document.head.appendChild(style);
  document.body.appendChild(panel);
  document.body.appendChild(launcher);
})();`);
};

//Helper method that returns the (first) value of a parameter if any was given.
const first = (param: unknown): string | undefined => {
  if (Array.isArray(param)) {
    return first(param[0]);
  } else if (typeof param === "string") {
    return param;
  } else {
    return undefined;
  }
};
