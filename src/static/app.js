/**
 * UDL Study Interface - Client Application
 * Vanilla JS, zero build-step, strictly same-origin API, in-memory keys only.
 */

(function () {
  'use strict';

  // =========================================================================
  // Ephemeral In-Memory State (NEVER persisted to storage, disk, or logs)
  // =========================================================================
  const state = {
    // Security: held only in JS runtime memory
    apiKey: '',
    provider: 'groq',
    model: '',
    customBaseUrl: '',
    chatHistory: [], // [{role: 'user'|'assistant', content: string}]

    // Query parameters
    mode: 'hybrid', // 'hybrid' | 'lexical' | 'semantic'
    modality: 'all', // 'all' | 'text' | 'code' | 'figure' | 'equation' | 'exercise'
    chapter: '', // '' for all
    topK: 6,

    // Store / Build Status
    activeBuildId: null,
    storeStatus: null,
    isBuildingIndex: false,

    // Active Streaming
    activeAbortController: null,
    isStreaming: false,

    // Review Queue
    pendingReviews: [],
    activeReviewChunk: null,

    // Active Inspector Chunk
    activeInspectorChunk: null
  };

  // Provider model presets
  const PROVIDER_DEFAULTS = {
    groq: '',
    gemini: '',
    openai_compat: '',
    ollama: ''
  };

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const dom = {
    // Header
    btnToggleSidebar: document.getElementById('btn-toggle-sidebar'),
    btnToggleRail: document.getElementById('btn-toggle-rail'),
    btnNavReviews: document.getElementById('btn-nav-reviews'),
    navReviewsBadge: document.getElementById('nav-reviews-badge'),
    tabReviewsBadge: document.getElementById('tab-reviews-badge'),
    indexStatusIndicator: document.getElementById('index-status-indicator'),

    // Sidebar
    sidebar: document.getElementById('sidebar'),
    selectProvider: document.getElementById('select-provider'),
    inputModel: document.getElementById('input-model'),
    inputApiKey: document.getElementById('input-api-key'),
    btnToggleKeyVisibility: document.getElementById('btn-toggle-key-visibility'),
    inputBaseUrl: document.getElementById('input-base-url'),
    groupBaseUrl: document.getElementById('group-base-url'),
    lblActiveBuild: document.getElementById('lbl-active-build'),
    lblChunkCount: document.getElementById('lbl-chunk-count'),
    lblDenseStatus: document.getElementById('lbl-dense-status'),
    chkSemanticBuild: document.getElementById('chk-semantic-build'),
    chkOfflineBuild: document.getElementById('chk-offline-build'),
    btnBuildIndex: document.getElementById('btn-build-index'),
    buildProgressMsg: document.getElementById('build-progress-msg'),
    chapterNavList: document.getElementById('chapter-nav-list'),

    // Workspace Tabs
    tabBtnAsk: document.getElementById('tab-btn-ask'),
    tabBtnSearch: document.getElementById('tab-btn-search'),
    tabBtnReviews: document.getElementById('tab-btn-reviews'),
    viewAsk: document.getElementById('view-ask'),
    viewSearch: document.getElementById('view-search'),
    viewReviews: document.getElementById('view-reviews'),

    // Filters
    selectMode: document.getElementById('select-mode'),
    selectModality: document.getElementById('select-modality'),
    selectChapterFilter: document.getElementById('select-chapter-filter'),
    inputTopK: document.getElementById('input-top-k'),
    lblTopKVal: document.getElementById('lbl-top-k-val'),

    // Chat / Ask View
    chatMessagesContainer: document.getElementById('chat-messages-container'),
    chatEmptyState: document.getElementById('chat-empty-state'),
    bannerChatAlert: document.getElementById('banner-chat-alert'),
    chatPromptInput: document.getElementById('chat-prompt-input'),
    btnSendAsk: document.getElementById('btn-send-ask'),
    btnStopStreaming: document.getElementById('btn-stop-streaming'),
    btnClearChat: document.getElementById('btn-clear-chat'),

    // Search View
    searchQueryInput: document.getElementById('search-query-input'),
    btnRunSearch: document.getElementById('btn-run-search'),
    searchResultsContainer: document.getElementById('search-results-container'),
    searchMetaBar: document.getElementById('search-meta-bar'),
    searchMetaInfo: document.getElementById('search-meta-info'),
    searchMetaWarnings: document.getElementById('search-meta-warnings'),
    searchEmptyState: document.getElementById('search-empty-state'),
    searchResultsList: document.getElementById('search-results-list'),

    // Reviews View
    reviewsQueueList: document.getElementById('reviews-queue-list'),
    btnRefreshReviews: document.getElementById('btn-refresh-reviews'),
    reviewWorkbenchEmpty: document.getElementById('review-workbench-empty'),
    reviewWorkbenchCard: document.getElementById('review-workbench-card'),
    reviewChunkTitle: document.getElementById('review-chunk-title'),
    reviewChunkLocator: document.getElementById('review-chunk-locator'),
    reviewStatusTag: document.getElementById('review-status-tag'),
    reviewCropImg: document.getElementById('review-crop-img'),
    reviewCropFallback: document.getElementById('review-crop-fallback'),
    reviewMathPreview: document.getElementById('review-math-preview'),
    reviewLatexInput: document.getElementById('review-latex-input'),
    reviewReviewerName: document.getElementById('review-reviewer-name'),
    btnDecisionPending: document.getElementById('btn-decision-pending'),
    btnDecisionVerified: document.getElementById('btn-decision-verified'),
    reviewSubmitStatus: document.getElementById('review-submit-status'),

    // Detail Rail
    detailRail: document.getElementById('detail-rail'),
    btnCloseRail: document.getElementById('btn-close-rail'),
    railChunkLocator: document.getElementById('rail-chunk-locator'),
    railEmptyState: document.getElementById('rail-empty-state'),
    railChunkCard: document.getElementById('rail-chunk-card'),
    railMetaChunkId: document.getElementById('rail-meta-chunk-id'),
    railMetaSourceId: document.getElementById('rail-meta-source-id'),
    railMetaContentType: document.getElementById('rail-meta-content-type'),
    railMetaLocation: document.getElementById('rail-meta-location'),
    railMetaReviewStatus: document.getElementById('rail-meta-review-status'),
    railMetaBuildId: document.getElementById('rail-meta-build-id'),
    railAssetContainer: document.getElementById('rail-asset-container'),
    railAssetImg: document.getElementById('rail-asset-img'),
    railVerbatimContainer: document.getElementById('rail-verbatim-container'),
    railTranscriptionSection: document.getElementById('rail-transcription-section'),
    railTranscription: document.getElementById('rail-transcription'),
    btnCopyVerbatim: document.getElementById('btn-copy-verbatim'),
    railPdfContainer: document.getElementById('rail-pdf-container'),
    railPdfIframe: document.getElementById('rail-pdf-iframe'),
    railPdfOpenLink: document.getElementById('rail-pdf-open-link'),
    railNotebookContainer: document.getElementById('rail-notebook-container'),
    railNotebookOpenLink: document.getElementById('rail-notebook-open-link')
  };

  // =========================================================================
  // Math & Markdown Rendering Utilities
  // =========================================================================

  /**
   * Safe LaTeX rendering using pinned local KaTeX.
   * STRICT SECURITY: trust is ALWAYS false.
   */
  function renderLatexToHtml(latex, displayMode = false) {
    if (typeof window.katex === 'undefined') {
      const span = document.createElement('span');
      span.textContent = latex;
      return span.outerHTML;
    }
    try {
      return window.katex.renderToString(latex, {
        displayMode: displayMode,
        throwOnError: false,
        trust: false // NEVER allow \url or raw HTML inside math expressions
      });
    } catch (err) {
      const span = document.createElement('span');
      span.className = 'katex-error font-mono';
      span.textContent = latex;
      return span.outerHTML;
    }
  }

  /**
   * Pre-processes text to extract and format LaTeX delimiters ($$..$$ and $..$)
   * before Markdown parsing, then sanitizes the final markup with DOMPurify.
   */
  function renderFormattedContent(text) {
    if (!text) return '';

    // Protect verbatim code before math handling, then restore it for Markdown.
    const codeBlocks = [];
    let processed = text.replace(/```[\s\S]*?```|`[^`\n]*`/g, (code) => {
      const token = `UDLCODEPLACEHOLDER${codeBlocks.length}TOKEN`;
      codeBlocks.push(code);
      return token;
    });
    const mathBlocks = [];
    const mathToken = (latex, display) => {
      const token = `UDLMATHPLACEHOLDER${mathBlocks.length}TOKEN`;
      mathBlocks.push(renderLatexToHtml(latex.trim(), display));
      return token;
    };
    processed = processed.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g,
      (_, dollars, brackets) => mathToken(dollars ?? brackets, true));
    processed = processed.replace(/\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)/g,
      (_, dollars, parentheses) => mathToken(dollars ?? parentheses, false));
    processed = processed.replace(/UDLCODEPLACEHOLDER(\d+)TOKEN/g,
      (_, index) => codeBlocks[Number(index)]);

    // Convert markdown to HTML using marked
    let rawHtml = '';
    if (typeof window.marked !== 'undefined' && typeof window.marked.parse === 'function') {
      rawHtml = window.marked.parse(processed);
    } else {
      // Fallback: escaped plaintext with line breaks
      const div = document.createElement('div');
      div.textContent = processed;
      rawHtml = div.innerHTML.replace(/\n/g, '<br>');
    }

    rawHtml = rawHtml.replace(/UDLMATHPLACEHOLDER(\d+)TOKEN/g,
      (_, index) => mathBlocks[Number(index)]);

    // STRICT SANITIZATION: Clean with DOMPurify allowing KaTeX and Prism classes/tags
    if (typeof window.DOMPurify !== 'undefined') {
      return window.DOMPurify.sanitize(rawHtml, {
        ADD_TAGS: ['annotation', 'semantics', 'math', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac'],
        ADD_ATTR: ['aria-hidden', 'display', 'xmlns', 'viewBox']
      });
    }
    const fallback = document.createElement('div');
    fallback.textContent = text;
    return fallback.innerHTML.replace(/\n/g, '<br>');
  }

  // =========================================================================
  // API Client (Relative Same-Origin URLs)
  // =========================================================================

  async function apiFetch(endpoint, options = {}) {
    const url = endpoint;
    const defaultHeaders = {
      'Accept': 'application/json'
    };
    if (options.body && typeof options.body === 'string') {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    const mergedOptions = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {})
      }
    };

    let response;
    try {
      response = await fetch(url, mergedOptions);
    } catch (err) {
      throw {
        code: 'NETWORK_ERROR',
        message: 'Could not connect to tutor service on localhost. Please check server status.',
        retryable: true
      };
    }

    if (!response.ok) {
      let errPayload = null;
      try {
        const json = await response.json();
        if (json && json.error) {
          errPayload = json.error;
        } else {
          errPayload = json;
        }
      } catch (_) {
        errPayload = {
          code: `HTTP_${response.status}`,
          message: response.statusText || `Request failed with HTTP ${response.status}`,
          retryable: response.status >= 500
        };
      }
      throw errPayload;
    }

    return response;
  }

  // =========================================================================
  // Status & Index Management
  // =========================================================================

  async function fetchStatus() {
    try {
      const resp = await apiFetch('/api/status');
      const data = await resp.json();
      state.storeStatus = data;
      state.activeBuildId = data.build_id || data.active_build_id || null;

      updateStatusUI(data);
      checkPendingReviewsCount();
    } catch (err) {
      showIndexConnectionError(err);
    }
  }

  function updateStatusUI(statusData) {
    const chunkCount = statusData.counts?.chunks ?? (statusData.chunk_count || (statusData.chunks ? statusData.chunks.length : 0) || statusData.num_chunks || 0);
    const denseIndexed = statusData.dense_ready ?? (statusData.dense_indexed || statusData.semantic_indexed || (statusData.dense_vectors_count > 0));
    const buildId = statusData.build_id || statusData.active_build_id || 'None';
    const job = statusData.job || statusData.build_job || {};
    const isJobRunning = job.status === 'running' || job.running;
    dom.btnBuildIndex.disabled = Boolean(isJobRunning);

    dom.lblActiveBuild.textContent = buildId === 'None' ? buildId : buildId.slice(-8);
    dom.lblActiveBuild.title = buildId;
    dom.lblChunkCount.textContent = chunkCount.toLocaleString();
    dom.lblDenseStatus.textContent = denseIndexed ? 'Available (Dense)' : 'Lexical Only';

    const statusDot = dom.indexStatusIndicator.querySelector('.status-dot');
    const statusText = dom.indexStatusIndicator.querySelector('.status-text');

    if (isJobRunning) {
      statusDot.className = 'status-dot status-building';
      statusText.textContent = `Building index (${job.phase || job.stage || 'in progress'})...`;
      dom.buildProgressMsg.style.display = 'block';
      dom.buildProgressMsg.textContent = `Index build active: ${job.message || job.phase || 'Processing corpus...'}`;
      state.isBuildingIndex = true;
      scheduleStatusPoll();
    } else if (chunkCount === 0) {
      statusDot.className = 'status-dot status-empty';
      statusText.textContent = 'Empty Index (Build Required)';
      dom.buildProgressMsg.style.display = job.error ? 'block' : 'none';
      if (job.error) dom.buildProgressMsg.textContent = `Build failed: ${job.error.message}`;
      state.isBuildingIndex = false;
    } else {
      statusDot.className = 'status-dot status-ready';
      const pendingMath = statusData.counts?.equations_pending || 0;
      const pendingFigures = (statusData.counts?.figures_pending || 0) + (statusData.counts?.algorithms_pending || 0);
      const reviewNote = pendingMath || pendingFigures ? ` · Review pending: ${pendingMath} equations, ${pendingFigures} visuals` : '';
      const readinessLabel = statusData.release_ready ? 'Validated corpus' : 'Index ready';
      statusText.textContent = `${readinessLabel} (${chunkCount} chunks)${reviewNote}`;
      dom.indexStatusIndicator.title = statusData.release_ready
        ? 'Source fidelity and held-out retrieval checks passed for this build.'
        : 'Search is available. Complete source reviews and retrieval validation before treating this build as release-validated.';
      dom.buildProgressMsg.style.display = job.error ? 'block' : 'none';
      if (job.error) dom.buildProgressMsg.textContent = `Build failed; previous index retained: ${job.error.message}`;
      state.isBuildingIndex = false;
    }
  }

  function showIndexConnectionError(err) {
    const statusDot = dom.indexStatusIndicator.querySelector('.status-dot');
    const statusText = dom.indexStatusIndicator.querySelector('.status-text');
    statusDot.className = 'status-dot status-empty';
    statusText.textContent = 'Store Offline / Not Started';
    dom.lblActiveBuild.textContent = 'Unreachable';
    dom.lblChunkCount.textContent = '0';
  }

  let statusPollTimer = null;
  function scheduleStatusPoll() {
    if (statusPollTimer) clearTimeout(statusPollTimer);
    statusPollTimer = setTimeout(() => {
      fetchStatus();
    }, 2500);
  }

  async function triggerBuildIndex() {
    if (state.isBuildingIndex) return;

    const semantic = dom.chkSemanticBuild.checked;
    const offline = dom.chkOfflineBuild.checked;

    dom.btnBuildIndex.disabled = true;
    dom.buildProgressMsg.style.display = 'block';
    dom.buildProgressMsg.textContent = 'Requesting index build from backend...';

    try {
      const resp = await apiFetch('/api/build', {
        method: 'POST',
        body: JSON.stringify({ semantic, offline })
      });
      const res = await resp.json();
      state.isBuildingIndex = true;
      dom.buildProgressMsg.textContent = 'Build job dispatched. Monitoring progress...';
      fetchStatus();
    } catch (err) {
      dom.btnBuildIndex.disabled = false;
      dom.buildProgressMsg.style.display = 'block';
      dom.buildProgressMsg.className = 'build-status-msg'
      dom.buildProgressMsg.style.backgroundColor = 'var(--danger-bg)';
      dom.buildProgressMsg.style.color = 'var(--danger-ink)';
      dom.buildProgressMsg.textContent = `Build failed: ${err.message || 'Unknown error'}`;
    }
  }

  // =========================================================================
  // Ask Tutor (SSE Streaming Generation)
  // =========================================================================

  async function handleSendAsk() {
    const query = dom.chatPromptInput.value.trim();
    if (!query || state.isStreaming) return;

    // Validate in-memory API key for external cloud providers
    const serverKeyConfigured = state.storeStatus?.providers?.some(p => p.id === state.provider && p.configured);
    if (state.provider !== 'ollama' && !state.apiKey && !serverKeyConfigured) {
      showChatAlert(`Please provide an API key in the left sidebar for provider: ${state.provider.toUpperCase()} (held in memory only).`);
      dom.inputApiKey.focus();
      return;
    }

    if (!state.model) {
      showChatAlert('Enter an exact model ID from your provider account or local Ollama installation.');
      dom.inputModel.focus();
      return;
    }

    clearChatAlert();

    // Hide empty state
    if (dom.chatEmptyState) {
      dom.chatEmptyState.style.display = 'none';
    }

    // Append user message
    appendUserMessage(query);
    dom.chatPromptInput.value = '';

    // Prepare assistant message shell
    const assistantMsgObj = appendAssistantMessageShell();

    // Prepare AbortController
    state.activeAbortController = new AbortController();
    state.isStreaming = true;
    toggleStreamingControls(true);

    const askPayload = {
      query: query,
      top_k: parseInt(state.topK, 10),
      mode: state.mode,
      modality: state.modality,
      chapter: state.chapter || null,
      provider: state.provider,
      model: state.model,
      api_key: state.apiKey,
      base_url: state.customBaseUrl || null,
      history: state.chatHistory.slice(-10) // Bound in-memory history to last 10 turns
    };

    let accumulatedText = '';
    let receivedSources = [];
    let receivedWarnings = [];

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(askPayload),
        signal: state.activeAbortController.signal
      });

      if (!response.ok) {
        let errJson = null;
        try {
          errJson = await response.json();
        } catch (_) {}
        const code = errJson?.error?.code || `HTTP_${response.status}`;
        const message = errJson?.error?.message || response.statusText || 'Inference request failed.';
        throw { code, message };
      }

      // Consume ReadableStream
      await consumeSSEStream(response, (eventType, data) => {
        if (eventType === 'sources') {
          // data is SearchResult
          if (data && data.hits) {
            receivedSources = data.hits;
            assistantMsgObj.updateSources(receivedSources, data.build_id);
          }
          if (data && data.warnings && data.warnings.length > 0) {
            assistantMsgObj.setWarnings(data.warnings);
          }
        } else if (eventType === 'delta') {
          if (data && typeof data.text === 'string') {
            accumulatedText += data.text;
            assistantMsgObj.updateDelta(accumulatedText);
          }
        } else if (eventType === 'done') {
          if (data && data.answer) {
            accumulatedText = data.answer;
          }
          if (data && data.sources) {
            receivedSources = data.sources;
          }
          if (data && data.warnings) {
            receivedWarnings = data.warnings;
          }
          assistantMsgObj.finalize(accumulatedText, receivedSources, receivedWarnings, data?.abstained);

          // Update in-memory history
          state.chatHistory.push({ role: 'user', content: query });
          state.chatHistory.push({ role: 'assistant', content: accumulatedText });
        } else if (eventType === 'error') {
          assistantMsgObj.setError(data?.message || 'Generation error reported by tutor.');
        }
      }, state.activeAbortController.signal);

    } catch (err) {
      if (err.name === 'AbortError') {
        assistantMsgObj.setWarning('Generation stopped by user.');
        if (accumulatedText) {
          assistantMsgObj.finalize(accumulatedText, receivedSources, receivedWarnings, false);
          state.chatHistory.push({ role: 'user', content: query });
          state.chatHistory.push({ role: 'assistant', content: accumulatedText });
        }
      } else {
        handleStreamingError(assistantMsgObj, err);
      }
    } finally {
      state.isStreaming = false;
      state.activeAbortController = null;
      toggleStreamingControls(false);
      scrollChatToBottom();
    }
  }

  function handleStreamingError(msgObj, err) {
    let friendlyMsg = err.message || 'Stream connection interrupted.';
    if (err.code === 'EMPTY_DATABASE' || err.code === 'EMPTY_STORE') {
      friendlyMsg = 'Corpus index is empty. Please click "Build Local Index" in the sidebar first.';
    } else if (err.code === 'EMBEDDINGS_MISSING' || (err.message && err.message.toLowerCase().includes('semantic'))) {
      friendlyMsg = 'Semantic dense vectors are missing. Switch to "Lexical" retrieval mode or rebuild index with dense embeddings.';
    } else if (err.code === 'AUTH_ERROR' || err.code === 'HTTP_401' || err.code === 'HTTP_403') {
      friendlyMsg = 'Authentication failed with model provider. Please check your in-memory API key in the sidebar.';
    } else if (err.code === 'NETWORK_ERROR') {
      friendlyMsg = 'Failed to connect to backend server. Ensure the FastAPI tutor server is active.';
    }
    msgObj.setError(friendlyMsg);
  }

  /**
   * Robust SSE stream parser consuming ReadableStream.
   * Handles multi-byte UTF-8, split lines, and custom event/data formats.
   */
  async function consumeSSEStream(response, onEvent, signal) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let currentEvent = 'message';
    let currentDataLines = [];

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r\n|\r|\n/);
        // Leave unfinished line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line === '') {
            // End of event block
            if (currentDataLines.length > 0) {
              const fullDataStr = currentDataLines.join('\n');
              let parsedData = fullDataStr;
              try {
                parsedData = JSON.parse(fullDataStr);
              } catch (_) {}
              onEvent(currentEvent, parsedData);
            }
            currentEvent = 'message';
            currentDataLines = [];
          } else if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            currentDataLines.push(line.slice(5).trimStart());
          }
        }
      }

      // Flush remainder
      if (buffer.trim()) {
        if (buffer.startsWith('data:')) {
          currentDataLines.push(buffer.slice(5).trimStart());
        }
      }
      if (currentDataLines.length > 0) {
        const fullDataStr = currentDataLines.join('\n');
        let parsedData = fullDataStr;
        try {
          parsedData = JSON.parse(fullDataStr);
        } catch (_) {}
        onEvent(currentEvent, parsedData);
      }
    } finally {
      reader.releaseLock();
    }
  }

  function toggleStreamingControls(isStreaming) {
    dom.btnSendAsk.style.display = isStreaming ? 'none' : 'inline-flex';
    dom.btnStopStreaming.style.display = isStreaming ? 'inline-flex' : 'none';
    dom.chatPromptInput.disabled = isStreaming;
  }

  function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message chat-message-user';

    const bubble = document.createElement('div');
    bubble.className = 'user-bubble';
    bubble.textContent = text; // Plain textContent for strict security

    msgDiv.appendChild(bubble);
    dom.chatMessagesContainer.appendChild(msgDiv);
    scrollChatToBottom();
  }

  function appendAssistantMessageShell() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message chat-message-assistant';

    const bubble = document.createElement('div');
    bubble.className = 'assistant-bubble';

    const warningBanner = document.createElement('div');
    warningBanner.className = 'assistant-warning-banner';
    warningBanner.style.display = 'none';

    const abstainedBanner = document.createElement('div');
    abstainedBanner.className = 'assistant-abstained-banner';
    abstainedBanner.style.display = 'none';
    abstainedBanner.textContent = 'The indexed book and notebooks do not provide enough support for this answer.';

    const contentArea = document.createElement('div');
    contentArea.className = 'assistant-body-content';
    contentArea.innerHTML = '<span class="status-dot status-loading" style="display:inline-block; vertical-align:middle; margin-right:6px;"></span><em>Consulting textbook & notebooks...</em>';

    const sourcesSection = document.createElement('div');
    sourcesSection.className = 'message-sources-section';
    sourcesSection.style.display = 'none';
    sourcesSection.innerHTML = '<div class="sources-header">Cited book and notebook sources</div><div class="sources-chips-grid"></div>';

    bubble.appendChild(warningBanner);
    bubble.appendChild(abstainedBanner);
    bubble.appendChild(contentArea);
    bubble.appendChild(sourcesSection);
    msgDiv.appendChild(bubble);

    dom.chatMessagesContainer.appendChild(msgDiv);
    scrollChatToBottom();

    return {
      updateDelta: (fullText) => {
        contentArea.innerHTML = renderFormattedContent(fullText);
        scrollChatToBottom();
      },
      updateSources: (hits, buildId) => {
        if (!hits || hits.length === 0) return;
        sourcesSection.style.display = 'flex';
        const grid = sourcesSection.querySelector('.sources-chips-grid');
        grid.innerHTML = '';
        hits.forEach(hit => {
          const pill = createCitationPill(hit, buildId);
          grid.appendChild(pill);
        });
      },
      setWarnings: (warnings) => {
        if (warnings && warnings.length > 0) {
          warningBanner.style.display = 'flex';
          warningBanner.textContent = warnings.join('; ');
        }
      },
      setError: (errMsg) => {
        contentArea.innerHTML = `<div class="chat-alert-banner">${DOMPurify.sanitize(errMsg)}</div>`;
      },
      setWarning: (warnMsg) => {
        warningBanner.style.display = 'flex';
        warningBanner.textContent = warnMsg;
      },
      finalize: (fullText, sources, warnings, abstained) => {
        if (abstained) {
          abstainedBanner.style.display = 'block';
        }
        if (warnings && warnings.length > 0) {
          warningBanner.style.display = 'flex';
          warningBanner.textContent = warnings.join('; ');
        }
        contentArea.innerHTML = renderFormattedContent(fullText);
        if (sources && sources.length > 0) {
          sourcesSection.style.display = 'flex';
          const grid = sourcesSection.querySelector('.sources-chips-grid');
          grid.innerHTML = '';
          sources.forEach(hit => {
            grid.appendChild(createCitationPill(hit, hit.build_id || state.activeBuildId));
          });
        }
        scrollChatToBottom();
      }
    };
  }

  function createCitationPill(hit, buildId) {
    const pill = document.createElement('button');
    pill.className = 'citation-pill';
    pill.type = 'button';

    const locator = hit.citation || hit.locator || formatLocatorFromChunk(hit);
    pill.textContent = `[${locator}]`;
    pill.title = `Inspect chunk: ${hit.chunk_id || ''}`;

    pill.addEventListener('click', (e) => {
      e.preventDefault();
      inspectChunk(hit.chunk_id, hit.build_id || buildId || state.activeBuildId);
    });

    return pill;
  }

  function formatLocatorFromChunk(chunk) {
    const meta = chunk.metadata || {};
    if (chunk.source_type === 'book' || chunk.source_id === 'book') {
      let loc = `p. ${meta.pdf_page || meta.book_page || '?'}`;
      if (meta.section) loc = `§${meta.section}, ` + loc;
      if (meta.chapter) loc = `Chap ${meta.chapter}, ` + loc;
      return loc;
    } else if (chunk.source_type === 'notebook') {
      const nbName = (chunk.source_id || 'Notebook').replace('Notebooks/', '');
      return `${nbName} #${meta.cell_index ?? '?'}`;
    }
    return chunk.chunk_id || 'Source';
  }

  function scrollChatToBottom() {
    dom.chatMessagesContainer.scrollTop = dom.chatMessagesContainer.scrollHeight;
  }

  function showChatAlert(msg) {
    dom.bannerChatAlert.style.display = 'block';
    dom.bannerChatAlert.textContent = msg;
  }

  function clearChatAlert() {
    dom.bannerChatAlert.style.display = 'none';
    dom.bannerChatAlert.textContent = '';
  }

  function clearChat() {
    state.chatHistory = [];
    dom.chatMessagesContainer.innerHTML = '';
    if (dom.chatEmptyState) {
      dom.chatEmptyState.style.display = 'block';
      dom.chatMessagesContainer.appendChild(dom.chatEmptyState);
    }
  }

  // =========================================================================
  // Search Corpus Explorer
  // =========================================================================

  async function handleRunSearch() {
    const query = dom.searchQueryInput.value.trim();
    if (!query) return;

    dom.btnRunSearch.disabled = true;
    dom.searchEmptyState.style.display = 'none';
    dom.searchResultsList.innerHTML = '<div class="loading-spinner-card">Searching indexed textbook & notebooks...</div>';
    dom.searchMetaBar.style.display = 'none';

    const searchPayload = {
      query: query,
      top_k: parseInt(state.topK, 10),
      mode: state.mode,
      modality: state.modality,
      chapter: state.chapter || null
    };

    try {
      const resp = await apiFetch('/api/search', {
        method: 'POST',
        body: JSON.stringify(searchPayload)
      });
      const result = await resp.json();
      renderSearchResults(result);
    } catch (err) {
      dom.searchResultsList.innerHTML = `<div class="chat-alert-banner">Search failed: ${DOMPurify.sanitize(err.message || 'Error executing retrieval')}</div>`;
    } finally {
      dom.btnRunSearch.disabled = false;
    }
  }

  function renderSearchResults(result) {
    dom.searchResultsList.innerHTML = '';
    const hits = result.hits || [];

    dom.searchMetaBar.style.display = 'flex';
    dom.searchMetaInfo.textContent = `Found ${hits.length} matches · Mode: ${result.mode || state.mode} · Build: ${result.build_id || state.activeBuildId || 'N/A'}`;
    
    if (result.warnings && result.warnings.length > 0) {
      dom.searchMetaWarnings.textContent = `Warning: ${result.warnings.join('; ')}`;
    } else {
      dom.searchMetaWarnings.textContent = '';
    }

    if (hits.length === 0) {
      dom.searchResultsList.innerHTML = `
        <div class="empty-state-card">
          <p>No chunks matched your query with current filters.</p>
          <small class="text-muted">Try switching retrieval mode to "Hybrid" or expanding modality to "All".</small>
        </div>`;
      return;
    }

    hits.forEach((hit, index) => {
      const card = document.createElement('div');
      card.className = 'search-result-card';

      const header = document.createElement('div');
      header.className = 'search-card-header';

      const locator = document.createElement('div');
      locator.className = 'search-card-locator';
      locator.textContent = hit.locator || hit.citation || formatLocatorFromChunk(hit);

      const modalityBadge = document.createElement('span');
      const mod = hit.content_type || 'text';
      modalityBadge.className = `modality-badge modality-${mod}`;
      modalityBadge.textContent = mod;

      header.appendChild(locator);
      header.appendChild(modalityBadge);

      // Scores
      const scores = document.createElement('div');
      scores.className = 'search-card-scores';

      if (hit.retrieval_score != null && result.mode === 'hybrid') {
        scores.innerHTML += `<span class="score-chip">Fusion score: ${Number(hit.retrieval_score).toFixed(4)}</span>`;
      }
      if (hit.bm25_score != null) {
        scores.innerHTML += `<span class="score-chip">BM25: ${Number(hit.bm25_score).toFixed(2)}</span>`;
      }
      if (hit.semantic_score != null) {
        scores.innerHTML += `<span class="score-chip">Cosine: ${Number(hit.semantic_score).toFixed(3)}</span>`;
      }
      if (hit.build_id) {
        const buildChip = document.createElement('span');
        buildChip.className = 'score-chip font-mono';
        buildChip.textContent = `Build: ${hit.build_id}`;
        scores.appendChild(buildChip);
      }

      // Snippet
      const snippet = document.createElement('div');
      snippet.className = 'search-card-snippet';
      const rawText = hit.content || hit.raw_content || '';
      snippet.textContent = rawText.length > 320 ? rawText.slice(0, 320) + '...' : rawText;

      card.appendChild(header);
      card.appendChild(scores);
      card.appendChild(snippet);

      card.addEventListener('click', () => {
        inspectChunk(hit.chunk_id, hit.build_id || result.build_id || state.activeBuildId);
      });
      card.tabIndex = 0;
      card.setAttribute('role', 'button');
      card.setAttribute('aria-label', `Open source: ${hit.locator || hit.chunk_id}`);
      card.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          card.click();
        }
      });

      dom.searchResultsList.appendChild(card);
    });
  }

  // =========================================================================
  // Source Detail Rail & Asset Inspector
  // =========================================================================

  async function inspectChunk(chunkId, buildId) {
    if (!chunkId) return;

    openRail();
    dom.railEmptyState.style.display = 'none';
    dom.railChunkCard.style.display = 'none';
    dom.railChunkLocator.textContent = `Loading ${chunkId}...`;

    try {
      const qParams = buildId ? `?build_id=${encodeURIComponent(buildId)}` : '';
      const resp = await apiFetch(`/api/chunks/${encodeURIComponent(chunkId)}${qParams}`);
      const chunk = await resp.json();
      state.activeInspectorChunk = chunk;
      renderInspectorChunk(chunk, buildId);
    } catch (err) {
      dom.railChunkLocator.textContent = 'Error loading chunk';
      dom.railEmptyState.style.display = 'block';
      dom.railEmptyState.innerHTML = `<div class="chat-alert-banner">Could not fetch chunk details: ${DOMPurify.sanitize(err.message || '404 Not Found')}</div>`;
    }
  }

  function renderInspectorChunk(chunk, buildId) {
    const meta = chunk.metadata || {};
    const bId = buildId || chunk.build_id || state.activeBuildId || '';

    dom.railChunkCard.style.display = 'flex';
    dom.railChunkLocator.textContent = formatLocatorFromChunk(chunk);

    // Meta chips
    dom.railMetaChunkId.textContent = chunk.chunk_id || 'N/A';
    dom.railMetaSourceId.textContent = chunk.source_id || 'N/A';
    dom.railMetaContentType.textContent = chunk.content_type || 'text';
    dom.railMetaBuildId.textContent = bId || 'N/A';
    dom.railMetaReviewStatus.textContent = meta.review_status ||
      (chunk.content_type === 'equation' ? 'pending' : meta.asset_path ? 'Source crop available' : 'Not required');

    let locationStr = '';
    if (meta.chapter) locationStr += `Chap ${meta.chapter} `;
    if (meta.section) locationStr += `§${meta.section} `;
    if (meta.pdf_page) locationStr += `PDF p.${meta.pdf_page} `;
    if (meta.cell_index != null) locationStr += `Cell [${meta.cell_index}]`;
    dom.railMetaLocation.textContent = locationStr || 'N/A';

    // Verbatim text
    const verbatim = chunk.raw_content || chunk.content || '';
    dom.railVerbatimContainer.textContent = verbatim; // textContent for security

    const verifiedMath = chunk.content_type === 'equation' && meta.review_status === 'verified';
    dom.railTranscriptionSection.style.display = verifiedMath ? 'block' : 'none';
    dom.railTranscription.textContent = '';
    if (verifiedMath) {
      const transcription = meta.transcription || chunk.content || '';
      if (window.katex) {
        try {
          katex.render(transcription, dom.railTranscription, {
            displayMode: true, throwOnError: true, trust: false
          });
        } catch (_) {
          dom.railTranscription.textContent = transcription;
        }
      } else {
        dom.railTranscription.textContent = transcription;
      }
    }

    // Figure or Equation Asset Crop
    if (meta.asset_path || chunk.content_type === 'figure' || chunk.content_type === 'equation') {
      dom.railAssetContainer.style.display = 'flex';
      const cropUrl = `/api/asset?chunk_id=${encodeURIComponent(chunk.chunk_id)}&build_id=${encodeURIComponent(bId)}`;
      dom.railAssetImg.src = cropUrl;
      dom.railAssetImg.onerror = () => {
        dom.railAssetContainer.style.display = 'none';
      };
    } else {
      dom.railAssetContainer.style.display = 'none';
      dom.railAssetImg.src = '';
    }

    // PDF Source Viewer
    if (chunk.source_type === 'book' || chunk.source_id === 'book') {
      dom.railPdfContainer.style.display = 'flex';
      const page = meta.pdf_page || meta.book_page || 1;
      const pdfUrl = `/api/source?source_id=book&build_id=${encodeURIComponent(bId)}#page=${page}`;
      dom.railPdfIframe.src = pdfUrl;
      dom.railPdfOpenLink.href = pdfUrl;
      dom.railNotebookContainer.style.display = 'none';
    } else if (chunk.source_type === 'notebook') {
      dom.railPdfContainer.style.display = 'none';
      dom.railNotebookContainer.style.display = 'flex';
      const nbUrl = `/api/source?source_id=${encodeURIComponent(chunk.source_id)}&build_id=${encodeURIComponent(bId)}`;
      dom.railNotebookOpenLink.href = nbUrl;
    } else {
      dom.railPdfContainer.style.display = 'none';
      dom.railNotebookContainer.style.display = 'none';
    }
  }

  function openRail() {
    dom.detailRail.classList.remove('collapsed');
  }

  function closeRail() {
    dom.detailRail.classList.add('collapsed');
  }

  function toggleRail() {
    dom.detailRail.classList.toggle('collapsed');
  }

  function toggleSidebar() {
    dom.sidebar.classList.toggle('collapsed');
  }

  // =========================================================================
  // Math Review Queue & Verification Workbench
  // =========================================================================

  async function checkPendingReviewsCount() {
    try {
      const resp = await apiFetch('/api/reviews');
      const data = await resp.json();
      const list = (data.reviews || data.pending || []).filter(item => item.metadata?.review_status !== 'verified');
      const count = list.length;
      if (count > 0) {
        dom.navReviewsBadge.style.display = 'inline-block';
        dom.navReviewsBadge.textContent = count;
        dom.tabReviewsBadge.style.display = 'inline-block';
        dom.tabReviewsBadge.textContent = count;
      } else {
        dom.navReviewsBadge.style.display = 'none';
        dom.tabReviewsBadge.style.display = 'none';
      }
      state.pendingReviews = list;
    } catch (_) {
      // Silent in background
    }
  }

  async function loadReviewQueue() {
    dom.reviewsQueueList.innerHTML = '<div class="loading-spinner-card">Loading review items...</div>';
    try {
      const resp = await apiFetch('/api/reviews');
      const data = await resp.json();
      const list = (data.reviews || data.pending || []).filter(item => item.metadata?.review_status !== 'verified');
      state.pendingReviews = list;
      renderReviewQueueList(list);
    } catch (err) {
      dom.reviewsQueueList.innerHTML = `<div class="chat-alert-banner">Failed loading reviews: ${DOMPurify.sanitize(err.message || 'Error')}</div>`;
    }
  }

  function renderReviewQueueList(list) {
    dom.reviewsQueueList.innerHTML = '';
    if (list.length === 0) {
      dom.reviewsQueueList.innerHTML = '<div class="empty-state-card" style="margin:20px 10px;"><p>No pending math equations in queue.</p></div>';
      return;
    }

    list.forEach(item => {
      const row = document.createElement('div');
      row.className = 'review-queue-item';
      if (state.activeReviewChunk && state.activeReviewChunk.chunk_id === item.chunk_id) {
        row.classList.add('active');
      }

      const title = document.createElement('div');
      title.className = 'review-item-title';
      title.textContent = item.metadata?.equation_number ? `Equation ${item.metadata.equation_number}` : (item.chunk_id || 'Equation');

      const meta = document.createElement('div');
      meta.className = 'review-item-meta';
      const chap = item.metadata?.chapter ? `Chap ${item.metadata.chapter}` : '';
      const pg = item.metadata?.pdf_page ? `p.${item.metadata.pdf_page}` : '';
      meta.textContent = `${chap} ${pg} · ${item.metadata?.review_status || 'pending'}`;

      row.appendChild(title);
      row.appendChild(meta);

      row.addEventListener('click', () => {
        document.querySelectorAll('.review-queue-item').forEach(el => el.classList.remove('active'));
        row.classList.add('active');
        selectReviewChunk(item);
      });

      dom.reviewsQueueList.appendChild(row);
    });

    // Auto-select first item if none active
    if (!state.activeReviewChunk && list.length > 0) {
      selectReviewChunk(list[0]);
      const first = dom.reviewsQueueList.querySelector('.review-queue-item');
      if (first) first.classList.add('active');
    }
  }

  function selectReviewChunk(chunk) {
    state.activeReviewChunk = chunk;
    dom.reviewWorkbenchEmpty.style.display = 'none';
    dom.reviewWorkbenchCard.style.display = 'flex';
    dom.reviewSubmitStatus.style.display = 'none';

    const meta = chunk.metadata || {};
    dom.reviewChunkTitle.textContent = meta.equation_number ? `Equation ${meta.equation_number}` : (chunk.chunk_id || 'Equation');
    dom.reviewChunkLocator.textContent = formatLocatorFromChunk(chunk);

    const isVerified = meta.review_status === 'verified';
    dom.reviewStatusTag.textContent = isVerified ? 'Verified' : 'Pending Verification';
    dom.reviewStatusTag.className = `review-status-tag ${isVerified ? 'verified' : ''}`;

    // Crop image
    const bId = chunk.build_id || state.activeBuildId || '';
    const cropUrl = `/api/asset?chunk_id=${encodeURIComponent(chunk.chunk_id)}&build_id=${encodeURIComponent(bId)}`;
    dom.reviewCropImg.src = cropUrl;
    dom.reviewCropFallback.style.display = 'none';
    dom.reviewCropImg.style.display = 'block';
    dom.reviewCropImg.onerror = () => {
      dom.reviewCropImg.style.display = 'none';
      dom.reviewCropFallback.style.display = 'block';
    };

    // LaTeX text
    const initialLatex = chunk.content || chunk.raw_content || '';
    dom.reviewLatexInput.value = initialLatex;
    updateMathLivePreview(initialLatex);
  }

  function updateMathLivePreview(latex) {
    dom.reviewMathPreview.innerHTML = renderLatexToHtml(latex.trim(), true);
  }

  async function submitReviewDecision(decision) {
    if (!state.activeReviewChunk) return;

    const chunkId = state.activeReviewChunk.chunk_id;
    const transcription = dom.reviewLatexInput.value.trim();
    const reviewer = dom.reviewReviewerName.value.trim() || 'Independent Reviewer';
    const buildId = state.activeReviewChunk.build_id || state.activeBuildId || null;

    dom.btnDecisionPending.disabled = true;
    dom.btnDecisionVerified.disabled = true;

    try {
      const resp = await apiFetch(`/api/reviews/${encodeURIComponent(chunkId)}`, {
        method: 'POST',
        body: JSON.stringify({
          transcription: transcription,
          reviewer: reviewer,
          decision: decision, // 'verified' | 'pending'
          build_id: buildId
        })
      });

      dom.reviewSubmitStatus.style.display = 'block';
      dom.reviewSubmitStatus.className = 'review-submit-status';
      dom.reviewSubmitStatus.style.backgroundColor = 'var(--success-bg)';
      dom.reviewSubmitStatus.style.color = 'var(--success-ink)';
      dom.reviewSubmitStatus.textContent = `Review recorded as "${decision.toUpperCase()}". Queue updating...`;

      // Refresh review queue
      await loadReviewQueue();
      await checkPendingReviewsCount();
    } catch (err) {
      dom.reviewSubmitStatus.style.display = 'block';
      dom.reviewSubmitStatus.className = 'review-submit-status';
      dom.reviewSubmitStatus.style.backgroundColor = 'var(--danger-bg)';
      dom.reviewSubmitStatus.style.color = 'var(--danger-ink)';
      dom.reviewSubmitStatus.textContent = `Failed recording review: ${err.message || 'Server error'}`;
    } finally {
      dom.btnDecisionPending.disabled = false;
      dom.btnDecisionVerified.disabled = false;
    }
  }

  // =========================================================================
  // Tab Switching
  // =========================================================================

  function switchWorkspaceTab(target) {
    const tabs = [
      { btn: dom.tabBtnAsk, view: dom.viewAsk, name: 'ask' },
      { btn: dom.tabBtnSearch, view: dom.viewSearch, name: 'search' },
      { btn: dom.tabBtnReviews, view: dom.viewReviews, name: 'reviews' }
    ];

    tabs.forEach(t => {
      const isActive = t.name === target;
      t.btn.classList.toggle('active', isActive);
      t.btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      t.view.classList.toggle('active', isActive);
    });

    if (target === 'reviews') {
      loadReviewQueue();
    }
  }

  // =========================================================================
  // Event Listeners & Initialization
  // =========================================================================

  function bindEvents() {
    // Navigation / Header
    dom.btnToggleSidebar.addEventListener('click', toggleSidebar);
    dom.btnToggleRail.addEventListener('click', toggleRail);
    dom.btnCloseRail.addEventListener('click', closeRail);

    dom.btnNavReviews.addEventListener('click', () => switchWorkspaceTab('reviews'));

    // Workspace Tabs
    dom.tabBtnAsk.addEventListener('click', () => switchWorkspaceTab('ask'));
    dom.tabBtnSearch.addEventListener('click', () => switchWorkspaceTab('search'));
    dom.tabBtnReviews.addEventListener('click', () => switchWorkspaceTab('reviews'));

    // Provider select
    dom.selectProvider.addEventListener('change', (e) => {
      state.provider = e.target.value;
      if (PROVIDER_DEFAULTS[state.provider]) {
        state.model = PROVIDER_DEFAULTS[state.provider];
        dom.inputModel.value = state.model;
      }
      dom.groupBaseUrl.style.display = (state.provider === 'ollama' || state.provider === 'openai_compat') ? 'flex' : 'none';
    });

    // Model input
    dom.inputModel.addEventListener('input', (e) => {
      state.model = e.target.value.trim();
    });

    // API Key input (Held strictly in JS memory)
    dom.inputApiKey.addEventListener('input', (e) => {
      state.apiKey = e.target.value;
    });

    dom.btnToggleKeyVisibility.addEventListener('click', () => {
      const current = dom.inputApiKey.type;
      dom.inputApiKey.type = current === 'password' ? 'text' : 'password';
    });

    // Base URL input
    dom.inputBaseUrl.addEventListener('input', (e) => {
      state.customBaseUrl = e.target.value.trim();
    });

    // Build Index
    dom.btnBuildIndex.addEventListener('click', triggerBuildIndex);

    // Filters
    dom.selectMode.addEventListener('change', (e) => {
      state.mode = e.target.value;
    });

    dom.selectModality.addEventListener('change', (e) => {
      state.modality = e.target.value;
    });

    dom.selectChapterFilter.addEventListener('change', (e) => {
      setChapterFilter(e.target.value);
    });

    dom.inputTopK.addEventListener('input', (e) => {
      state.topK = e.target.value;
      dom.lblTopKVal.textContent = state.topK;
    });

    // Chapter nav list clicks
    dom.chapterNavList.addEventListener('click', (e) => {
      const li = e.target.closest('.chapter-nav-item');
      if (!li) return;
      const chap = li.getAttribute('data-chapter') || '';
      setChapterFilter(chap);
    });

    // Ask Tutor
    dom.btnSendAsk.addEventListener('click', handleSendAsk);
    dom.chatPromptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendAsk();
      }
    });

    dom.btnStopStreaming.addEventListener('click', () => {
      if (state.activeAbortController) {
        state.activeAbortController.abort();
      }
    });

    dom.btnClearChat.addEventListener('click', clearChat);

    // Sample query chips
    document.querySelectorAll('.sample-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const q = chip.getAttribute('data-query');
        if (q) {
          dom.chatPromptInput.value = q;
          dom.chatPromptInput.focus();
        }
      });
    });

    // Search Explorer
    dom.btnRunSearch.addEventListener('click', handleRunSearch);
    dom.searchQueryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleRunSearch();
      }
    });

    // Copy Verbatim
    dom.btnCopyVerbatim.addEventListener('click', () => {
      const text = dom.railVerbatimContainer.textContent;
      if (navigator.clipboard && text) {
        navigator.clipboard.writeText(text).then(() => {
          dom.btnCopyVerbatim.textContent = 'Copied!';
          setTimeout(() => { dom.btnCopyVerbatim.textContent = 'Copy'; }, 1500);
        });
      }
    });

    // Review Workbench
    dom.btnRefreshReviews.addEventListener('click', loadReviewQueue);
    dom.reviewLatexInput.addEventListener('input', (e) => {
      updateMathLivePreview(e.target.value);
    });
    dom.btnDecisionVerified.addEventListener('click', () => submitReviewDecision('verified'));
    dom.btnDecisionPending.addEventListener('click', () => submitReviewDecision('pending'));
  }

  function setChapterFilter(chapter) {
    state.chapter = chapter;
    dom.selectChapterFilter.value = chapter;

    document.querySelectorAll('.chapter-nav-item').forEach(item => {
      const itemChap = item.getAttribute('data-chapter') || '';
      item.classList.toggle('active', itemChap === chapter);
    });
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  function init() {
    bindEvents();
    fetchStatus();

    // Responsive initial states: collapse rail on smaller screens
    if (window.innerWidth < 1200) {
      closeRail();
    }
  }

  // Bootstrap when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  if (window.innerWidth <= 860) { dom.sidebar.classList.add('collapsed'); dom.detailRail.classList.add('collapsed'); }
})();
