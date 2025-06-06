document.addEventListener('DOMContentLoaded', () => {
    console.log('[NVK_CONSOLE_JS] Initializing Nurvek Vehicle Arm Interface...');
    const DEBUG_PREFIX = '[NVK_DEBUG_JS]';

    const eventLogElement = document.getElementById('event-log');
    const apiLogEndpoint = 'http://localhost:1242/api/v1/vehicle_events/log';
    const systemTimeElement = document.getElementById('system-time');
    const sessionIdElement = document.getElementById('session-id-footer');
    const logCountElement = document.getElementById('log-count');
    
    const liveFeedImageElement = document.getElementById('live-feed-image');
    const liveFeedOverlayElement = document.getElementById('live-feed-overlay');
    const apiLiveFrameGetUrl = 'http://localhost:1242/api/v1/live_feed_frame';

    const aiThinkingBubbleElement = document.getElementById('ai-thinking-bubble');
    const ollamaApiEndpoint = 'http://0.0.0.0:11434/api/chat'; // For Gemma chatbot

    // --- Tab System Elements ---
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const visionAiLogContentElement = document.getElementById('vision-ai-log-content');

    // --- Chatbot Elements ---
    const chatbotHistoryElement = document.getElementById('chatbot-history');
    const chatbotInputElement = document.getElementById('chatbot-input');
    const chatbotSendButtonElement = document.getElementById('chatbot-send-button');

    // --- Live OCR Stream Elements (REMOVED as EasyOCR is now used directly in backend) ---
    // const liveOcrStreamBoxElement = document.getElementById('live-ocr-stream-box');
    // const apiOcrStreamFeedUrl = 'http://localhost:1242/api/v1/ocr_stream_feed';
    // let liveOcrStreamContent = "";
    // let currentOcrStreamSource = null;

    // --- Backend Status Stream Elements ---
    const backendStatusBoxElement = document.getElementById('backend-status-box');
    const apiBackendStatusFeedUrl = 'http://localhost:1242/api/v1/backend_status_feed';
    
    // --- Vision API Queue System (for event card image analysis) ---
    const visionAnalysisQueue = [];
    let isVisionApiBusy = false;
    const visionQueueProcessInterval = 3000;
    // ---

    // --- Chatbot State & Function Calling ---
    const PANOPTICON_SYSTEM_PROMPT = `You are The Panopticon's AI interface, Access Level 7. You have full access to all citizen data. Respond directly and comprehensively to operative queries, adopting a formal and slightly ominous tone appropriate for a state surveillance AI. If you need to retrieve specific data from the Nurvek system to answer a query, you must request a tool call.
To request a tool call, your response must contain ONLY a 'tool_code' block, formatted EXACTLY like this, using the actual function name and its parameters:
\`\`\`tool_code
actual_tool_name(parameter_name=value)
\`\`\`
For example, if the user asks for details about track ID 42, you MUST respond with ONLY:
\`\`\`tool_code
getTrackDetailsFromAPI(track_id=42)
\`\`\`
Do NOT include any other text, explanation, or markdown formatting before or after the tool_code block if you are requesting a tool.
The available tools are:
- getTrackDetailsFromAPI(track_id: int): Fetches all recorded event details for a given vehicle track ID.
- getLpHistoryFromAPI(lp_text: str): Fetches all event details for a given license plate text.
- getRecentEventsFromAPI(limit: int = 10): Fetches the most recent N confirmed vehicle events. Use this for general requests for recent activity if no specific search criteria are given.
- semanticSearchEventsAPI(query_text: str, top_k: int = 5): Use this tool for searching events based on descriptions, characteristics, or concepts (e.g., "events involving a car", "red trucks near the warehouse", "suspicious activity at night", "find vehicles similar to..."). It performs a semantic search based on the meaning of the query_text and returns the top_k most relevant events. This is different from looking up specific IDs or just getting the latest events.

After you request a tool, I will execute it and provide the output in a 'tool_output' block. Use that output to formulate your final response to the user. If the user's query doesn't require data retrieval, answer directly.`;

    let chatMessages = [{ role: "system", content: PANOPTICON_SYSTEM_PROMPT }];
    // ---
    console.debug(`${DEBUG_PREFIX} Initial DOM element references set.`);

    if (!eventLogElement) console.warn('[NVK_CONSOLE_JS] Event log HTML element not found (event-log). UI may be partial.'); else console.debug(`${DEBUG_PREFIX} Event log element found.`);
    if (!liveFeedImageElement || !liveFeedOverlayElement) console.warn('[NVK_CONSOLE_JS] CRITICAL: Live feed image or SVG overlay element not found! ESP boxes disabled.'); else console.debug(`${DEBUG_PREFIX} Live feed elements found.`);
    if (!aiThinkingBubbleElement) console.warn('[NVK_CONSOLE_JS] AI thinking bubble element not found! Vision analysis UI feedback disabled.'); else console.debug(`${DEBUG_PREFIX} AI thinking bubble element found.`);
    if (!visionAiLogContentElement) console.warn('[NVK_CONSOLE_JS] Vision AI log content element not found!'); else console.debug(`${DEBUG_PREFIX} Vision AI log content element found.`);
    if (!chatbotHistoryElement || !chatbotInputElement || !chatbotSendButtonElement) console.warn('[NVK_CONSOLE_JS] Chatbot interface elements not found!'); else console.debug(`${DEBUG_PREFIX} Chatbot interface elements found.`);
    // if (!liveOcrStreamBoxElement) console.warn('[NVK_CONSOLE_JS] Live OCR Stream Box element not found!'); else console.debug(`${DEBUG_PREFIX} Live OCR Stream Box element found.`); // Removed
    if (!backendStatusBoxElement) console.warn('[NVK_CONSOLE_JS] Backend Status Box element not found!'); else console.debug(`${DEBUG_PREFIX} Backend Status Box element found.`);

    if (sessionIdElement) {
        sessionIdElement.textContent = Math.random().toString(36).substring(2, 10).toUpperCase();
        console.debug(`${DEBUG_PREFIX} Session ID set: ${sessionIdElement.textContent}`);
    }

    function updateSystemTime() {
        if (systemTimeElement) {
            const now = new Date();
            systemTimeElement.textContent = `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}-${String(now.getUTCDate()).padStart(2, '0')} ${String(now.getUTCHours()).padStart(2, '0')}:${String(now.getUTCMinutes()).padStart(2, '0')}:${String(now.getUTCSeconds()).padStart(2, '0')} UTC`;
        }
    }
    updateSystemTime();
    setInterval(updateSystemTime, 1000);

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTabPaneId = button.getAttribute('data-tab');
            console.debug(`${DEBUG_PREFIX} Tab button clicked. Target pane ID: ${targetTabPaneId}`);
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            button.classList.add('active');
            document.getElementById(targetTabPaneId).classList.add('active');
            console.debug(`${DEBUG_PREFIX} Tab ${targetTabPaneId} activated.`);
        });
    });

    // --- JavaScript "Tools" for Gemma ---
    async function getTrackDetailsFromAPI(track_id) {
        console.debug(`${DEBUG_PREFIX} JS Tool: getTrackDetailsFromAPI called with track_id: ${track_id}`);
        try {
            const response = await fetch(`http://localhost:1242/api/v1/track_details/${track_id}`);
            if (!response.ok) {
                console.error(`${DEBUG_PREFIX} API error in getTrackDetailsFromAPI: ${response.status}`);
                return { error: `API error: ${response.status} ${response.statusText}` };
            }
            const data = await response.json();
            console.debug(`${DEBUG_PREFIX} JS Tool: getTrackDetailsFromAPI received:`, data);
            return data.length > 0 ? data : { info: "No details found for this track ID."};
        } catch (error) {
            console.error(`${DEBUG_PREFIX} Fetch error in getTrackDetailsFromAPI:`, error);
            return { error: `Failed to fetch track details: ${error.message}` };
        }
    }

    async function semanticSearchEventsAPI(query_text, top_k = 5) {
        console.debug(`${DEBUG_PREFIX} JS Tool: semanticSearchEventsAPI called with query: "${query_text}", top_k: ${top_k}`);
        try {
            const response = await fetch(`http://localhost:1242/api/v1/events/semantic_search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query_text: query_text, top_k: top_k })
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                console.error(`${DEBUG_PREFIX} API error in semanticSearchEventsAPI: ${response.status}`, errorData);
                return { error: `API error: ${response.status} - ${errorData.detail || 'Unknown error'}` };
            }
            const data = await response.json();
            console.debug(`${DEBUG_PREFIX} JS Tool: semanticSearchEventsAPI received:`, data);
            return data.results && data.results.length > 0 ? data : { info: "No relevant events found for this query.", query_text: query_text };
        } catch (error) {
            console.error(`${DEBUG_PREFIX} Fetch error in semanticSearchEventsAPI:`, error);
            return { error: `Failed to perform semantic search: ${error.message}` };
        }
    }
    // TODO: Implement getLpHistoryFromAPI and getRecentEventsFromAPI if needed, along with their FastAPI endpoints.

    function displayChatMessage(messageText, sender) {
        // ... (same as before)
        console.debug(`${DEBUG_PREFIX} displayChatMessage called. Sender: ${sender}, Message: "${messageText.substring(0, 50)}..."`);
        if (!chatbotHistoryElement) {
            console.warn(`${DEBUG_PREFIX} chatbotHistoryElement not found in displayChatMessage.`);
            return;
        }
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        if (sender === 'USER') {
            messageElement.classList.add('user-message');
        } else if (sender === 'AI_INTERNAL') {
            messageElement.classList.add('internal-message');
        } else { // Default to AI
            messageElement.classList.add('ai-message');
        }
        
        const senderStrong = document.createElement('strong');
        senderStrong.textContent = sender === 'USER' ? 'OPERATIVE (LVL 7):' : 'PANOPTICON AI:';
        messageElement.appendChild(senderStrong);

        const contentP = document.createElement('p');
        contentP.textContent = messageText; // For non-streaming, or initial part of streaming
        messageElement.appendChild(contentP);
        
        chatbotHistoryElement.appendChild(messageElement);
        chatbotHistoryElement.scrollTop = chatbotHistoryElement.scrollHeight;
        return messageElement; 
    }

    async function handleSendQuery() {
        console.debug(`${DEBUG_PREFIX} handleSendQuery called.`);
        if (!chatbotInputElement || !chatbotSendButtonElement) return;
        const queryText = chatbotInputElement.value.trim();
        if (!queryText) return;

        displayChatMessage(queryText, "USER");
        // Add user message to chat history for Gemma
        const currentTurnMessages = [...chatMessages, { role: "user", content: queryText }];
        chatMessages.push({ role: "user", content: queryText }); // Persist user message

        chatbotInputElement.value = '';
        chatbotInputElement.disabled = true;
        chatbotSendButtonElement.disabled = true;
        chatbotSendButtonElement.textContent = 'PROCESSING...';

        // Initial call to Gemma
        let gemmaResponseText = await callGemmaAndProcessTools(currentTurnMessages);
        
        // Display final Gemma response
        const aiMessageElement = displayChatMessage("Thinking...", "AI");
        aiMessageElement.querySelector('p').textContent = gemmaResponseText;
        chatMessages.push({ role: "assistant", content: gemmaResponseText }); // Persist final AI response

        chatbotInputElement.disabled = false;
        chatbotSendButtonElement.disabled = false;
        chatbotSendButtonElement.textContent = 'TRANSMIT QUERY';
        chatbotInputElement.focus();
        console.debug(`${DEBUG_PREFIX} handleSendQuery completed.`);
    }
    
    async function callGemmaAndProcessTools(currentMessages) {
        let gemmaResponseText = await streamGemmaResponse(currentMessages, null); // No initial AI message element for first call

        // Regex to find tool_code block, allowing for variations in spacing and optional backticks
        const toolCallMatch = gemmaResponseText.match(/(?:```tool_code|tool_code)\s*([\s\S]*?)(?:\s*```|$)/);

        if (toolCallMatch) {
            const toolCodeContent = toolCallMatch[1].trim();
            console.debug(`${DEBUG_PREFIX} Gemma requested tool call content: ${toolCodeContent}`);
            displayChatMessage(`[Accessing Archives: ${toolCodeContent}]`, "AI_INTERNAL");

            let toolOutput = { error: "Failed to parse or execute tool call." };
            try {
                // Attempt to parse Gemma's generic functionName(param1="actualTool", param2=value)
                // Or the direct actualTool(param=value)
                let actualToolName = "";
                let actualArgs = {};

                // Try parsing functionName(param1="toolName", param2=value, ...)
                const genericFuncMatch = toolCodeContent.match(/(\w+)\s*\((.*)\)/);
                if (genericFuncMatch) {
                    const outerFuncName = genericFuncMatch[1]; // e.g., "functionName" or the actual tool name
                    const allArgsStr = genericFuncMatch[2];

                    // Extract key-value pairs, tolerant to some formatting
                    const argPattern = /(\w+)\s*=\s*(?:"([^"]*)"|(\d+\.?\d*|\w+))/g;
                    let match;
                    const parsedArgs = {};
                    while ((match = argPattern.exec(allArgsStr)) !== null) {
                        // match[1] is the key
                        // match[2] is double-quoted string, match[3] is single-quoted, match[4] is number/boolean/null/identifier
                        let value = match[2] !== undefined ? match[2] :
                                    match[3] !== undefined ? match[3] :
                                    match[4];
                        if (typeof value === 'string') {
                            if (!isNaN(parseFloat(value)) && isFinite(value)) value = parseFloat(value);
                            else if (value === 'true') value = true;
                            else if (value === 'false') value = false;
                            else if (value === 'null') value = null;
                        }
                        parsedArgs[match[1]] = value;
                    }
                    
                    console.debug(`${DEBUG_PREFIX} Parsed args from tool code:`, parsedArgs);

                    // Determine actual tool name and arguments
                    if (outerFuncName === "functionName" && parsedArgs.param1 && typeof parsedArgs.param1 === 'string') {
                        actualToolName = parsedArgs.param1;
                        // Map param2, param3 etc. to specific argument names for the actual tool
                        if (actualToolName === "getTrackDetailsFromAPI" && parsedArgs.param2 !== undefined) {
                            actualArgs.track_id = parseInt(parsedArgs.param2, 10);
                        } else if (actualToolName === "getLpHistoryFromAPI" && parsedArgs.param2 !== undefined) {
                            actualArgs.lp_text = String(parsedArgs.param2);
                        } else if (actualToolName === "getRecentEventsFromAPI" && parsedArgs.param2 !== undefined) {
                            actualArgs.limit = parseInt(parsedArgs.param2, 10);
                        } else if (actualToolName === "semanticSearchEventsAPI") {
                            if (parsedArgs.param2 !== undefined) actualArgs.query_text = String(parsedArgs.param2);
                            if (parsedArgs.param3 !== undefined) actualArgs.top_k = parseInt(parsedArgs.param3, 10);
                            else actualArgs.top_k = 5; // Default if not provided
                        }
                        // Add more mappings if Gemma uses param3, param4 for other tools/args
                    } else { // Assume outerFuncName is the actual tool name
                        actualToolName = outerFuncName;
                        actualArgs = parsedArgs; // Use parsedArgs directly
                    }

                    console.debug(`${DEBUG_PREFIX} Determined actual tool: ${actualToolName}, actualArgs:`, actualArgs);

                    // Execute the determined tool
                    if (actualToolName === "getTrackDetailsFromAPI" && actualArgs.track_id !== undefined) {
                        toolOutput = await getTrackDetailsFromAPI(parseInt(actualArgs.track_id, 10));
                    } else if (actualToolName === "getLpHistoryFromAPI" && actualArgs.lp_text !== undefined) {
                        toolOutput = { error: "getLpHistoryFromAPI tool not yet implemented in JS." }; // Placeholder
                    } else if (actualToolName === "getRecentEventsFromAPI") {
                        toolOutput = { error: "getRecentEventsFromAPI tool not yet implemented in JS." }; // Placeholder
                    } else if (actualToolName === "semanticSearchEventsAPI" && actualArgs.query_text !== undefined) {
                        toolOutput = await semanticSearchEventsAPI(actualArgs.query_text, actualArgs.top_k);
                    } else {
                        toolOutput = { error: `Unknown or improperly called tool: '${actualToolName}' with args ${JSON.stringify(actualArgs)}` };
                    }
                } else {
                     toolOutput = { error: `Could not parse tool call structure from: ${toolCodeContent}` };
                }
            } catch (e) {
                console.error(`${DEBUG_PREFIX} Error executing JS tool:`, e);
                toolOutput = { error: `Error during tool execution: ${e.message}` };
            }

            const toolOutputString = `\`\`\`tool_output\n${JSON.stringify(toolOutput, null, 2)}\n\`\`\``;
            console.debug(`${DEBUG_PREFIX} Tool output: ${toolOutputString}`);
            // Displaying the raw tool output might be too verbose for "AI_INTERNAL" if it's large.
            // We can display a summary or just log it. For now, let's display a shorter message.
            displayChatMessage(`[Tool Execution Result Received. Formulating response...]`, "AI_INTERNAL");
            console.debug(`${DEBUG_PREFIX} Full tool output for Gemma: ${JSON.stringify(toolOutput, null, 2)}`);


            // Add Gemma's request and tool output to history for the next call
            const messagesForNextTurn = [
                ...currentMessages, // Includes original user query and system prompt
                { role: "assistant", content: gemmaResponseText }, // Gemma's response that included the tool_code
                { role: "user", content: toolOutputString } // User role for tool output as per Gemma's expectation
            ];
            chatMessages.push({ role: "assistant", content: gemmaResponseText });
            chatMessages.push({ role: "user", content: toolOutputString });


            // Second call to Gemma with tool output
            gemmaResponseText = await streamGemmaResponse(messagesForNextTurn, null);
        }
        return gemmaResponseText;
    }

    async function streamGemmaResponse(messagesPayload, aiMessageElementToUpdate) {
        // If aiMessageElementToUpdate is null, it means it's an intermediate call (e.g., getting tool_code)
        // and we don't update a chat bubble live, but return the full text.
        // If it's provided, we stream into it. This function is now dual-purpose.

        let tempAiMessageElement = aiMessageElementToUpdate;
        let tempAiContentP = null;

        if (tempAiMessageElement) {
            tempAiContentP = tempAiMessageElement.querySelector('p');
            tempAiContentP.textContent = '';
        }
        
        let fullResponseText = "";
        console.debug(`${DEBUG_PREFIX} streamGemmaResponse called. Streaming: ${!!tempAiMessageElement}`);

        try {
            const requestPayload = { model: "gemma3:1b", messages: messagesPayload, stream: true };
            console.debug(`${DEBUG_PREFIX} Sending to Gemma API (streamGemmaResponse). Endpoint: ${ollamaApiEndpoint}, Payload:`, JSON.stringify(requestPayload, null, 2));
            
            const response = await fetch(ollamaApiEndpoint, { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload)
            });
            console.debug(`${DEBUG_PREFIX} Gemma API response status (streamGemmaResponse): ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`${DEBUG_PREFIX} Gemma API error (streamGemmaResponse): ${response.status} - ${errorText}`);
                fullResponseText = `PANOPTICON COMMS ERROR [${response.status}]: ${errorText}`;
                if (tempAiContentP) tempAiContentP.textContent = fullResponseText;
                return fullResponseText;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); 
                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const streamObj = JSON.parse(line);
                        if (streamObj.message && streamObj.message.content) {
                            const chunk = streamObj.message.content;
                            fullResponseText += chunk;
                            if (tempAiContentP) {
                                tempAiContentP.textContent = fullResponseText; 
                                chatbotHistoryElement.scrollTop = chatbotHistoryElement.scrollHeight;
                            }
                        }
                    } catch (e) { /* console.warn for parsing error */ }
                }
            }
            if (buffer.trim() !== '') { /* process final buffer */ 
                try {
                    const streamObj = JSON.parse(buffer);
                    if (streamObj.message && streamObj.message.content) {
                        fullResponseText += streamObj.message.content;
                        if (tempAiContentP) tempAiContentP.textContent = fullResponseText;
                    }
                } catch (e) { /* console.warn */ }
            }
        } catch (error) {
            console.error(`${DEBUG_PREFIX} Error in streamGemmaResponse:`, error);
            fullResponseText = `PANOPTICON OFFLINE // CONNECTION FAILED: ${error.message}`;
            if (tempAiContentP) tempAiContentP.textContent = fullResponseText;
        }
        
        if (fullResponseText.trim() === "" && tempAiContentP) {
            tempAiContentP.textContent = "PANOPTICON AI: No textual response generated.";
            fullResponseText = "PANOPTICON AI: No textual response generated.";
        }
        console.debug(`${DEBUG_PREFIX} streamGemmaResponse finished. Full text: "${fullResponseText.substring(0,100)}..."`);
        return fullResponseText;
    }


    if (chatbotSendButtonElement && chatbotInputElement) {
        chatbotSendButtonElement.addEventListener('click', () => {
            console.debug(`${DEBUG_PREFIX} Chatbot send button clicked.`);
            handleSendQuery();
        });
        chatbotInputElement.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                console.debug(`${DEBUG_PREFIX} Chatbot input Enter key pressed.`);
                event.preventDefault();
                handleSendQuery();
            }
        });
    }
    // --- End Chatbot Functions ---

    // --- Live OCR Stream (SSE) Handler (REMOVED as EasyOCR is now used directly in backend) ---
    // function setupLiveOcrStream() { ... }
    // if (liveOcrStreamBoxElement) setupLiveOcrStream(); // Removed
    // --- End Live OCR Stream Handler ---

    // --- Backend Status Stream (SSE) Handler ---
    function setupBackendStatusStream() {
        // ... (same as before)
        console.debug(`${DEBUG_PREFIX} Setting up Backend Status Stream (EventSource to ${apiBackendStatusFeedUrl})`);
        if (!backendStatusBoxElement) {
            console.warn(`${DEBUG_PREFIX} Backend Status Box element not found. SSE connection not started.`);
            return;
        }

        const eventSource = new EventSource(apiBackendStatusFeedUrl);
        backendStatusBoxElement.innerHTML = '<p class="placeholder-text">Connecting to backend status stream...</p>';
        let currentStatusContent = ""; 

        eventSource.onopen = () => {
            console.log(`${DEBUG_PREFIX} Backend Status EventSource connected.`);
            backendStatusBoxElement.innerHTML = '<p class="placeholder-text">Awaiting backend status updates...</p>';
            currentStatusContent = ""; 
        };

        eventSource.addEventListener('backend_status', (event) => {
            console.debug(`${DEBUG_PREFIX} SSE 'backend_status' received:`, event.data);
            try {
                const statusData = JSON.parse(event.data); 
                
                const timestamp = new Date(statusData.timestamp).toLocaleTimeString();
                let dataStr = "";
                if (statusData.data && Object.keys(statusData.data).length > 0) {
                    dataStr = Object.entries(statusData.data)
                                  .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                                  .join(', ');
                }
                const statusLine = `[${timestamp}] ${statusData.source} - ${statusData.event_type}: ${dataStr || 'No specific data'}\n`;
                
                currentStatusContent = statusLine + currentStatusContent; 
                if (currentStatusContent.length > 5000) { 
                    currentStatusContent = currentStatusContent.substring(0, 5000) + "\n... (log truncated)";
                }
                backendStatusBoxElement.textContent = currentStatusContent;

            } catch (e) {
                console.error(`${DEBUG_PREFIX} Error processing SSE backend_status data:`, e, "Raw data:", event.data);
                backendStatusBoxElement.textContent = currentStatusContent + "\n[STATUS STREAM ERROR]";
            }
        });

        eventSource.onerror = (error) => {
            console.error(`${DEBUG_PREFIX} Backend Status EventSource error:`, error);
            if (backendStatusBoxElement) backendStatusBoxElement.textContent = "Backend Status Stream DISCONNECTED. Check API server.";
            eventSource.close();
            setTimeout(setupBackendStatusStream, 5000); 
        };
    }
    if (backendStatusBoxElement) setupBackendStatusStream();
    // --- End Backend Status Stream Handler ---

    let displayedEventTimestamps = new Set();
    let totalEventsDisplayed = 0;

    async function analyzeEventImageWithVisionAPI(base64ImageData, eventCardElement, eventTimestamp) {
        // ... (same as before, using gemma3:4b for vision analysis on event cards)
        console.debug(`${DEBUG_PREFIX} analyzeEventImageWithVisionAPI called for event: ${eventTimestamp}`);
        const analysisDivId = `analysis-${eventTimestamp.replace(/[^a-zA-Z0-9]/g, "")}`;
        const analysisElement = eventCardElement.querySelector(`#${analysisDivId}`);
        const colorElement = eventCardElement.querySelector('.vehicle-color-ai');
        const plateTextSpan = eventCardElement.querySelector('.original-plate-text');

        if (!analysisElement) {
            console.warn(`${DEBUG_PREFIX} Analysis element not found in card for ${eventTimestamp}`);
            isVisionApiBusy = false; return;
        }
        
        if (aiThinkingBubbleElement) {
            aiThinkingBubbleElement.textContent = `AI: Verifying color/plate for event ${eventTimestamp.slice(-6)}... Queue: ${visionAnalysisQueue.length}`;
            aiThinkingBubbleElement.classList.add('visible');
        }
        analysisElement.innerHTML = `<p class="ai-analysis-status">NURVEK AI: Secondary Analysis In Progress...</p>`;

        const pureBase64 = base64ImageData.startsWith('data:image/jpeg;base64,') 
            ? base64ImageData.substring('data:image/jpeg;base64,'.length) 
            : base64ImageData;
        console.debug(`${DEBUG_PREFIX} Image data prepared for Vision API (length: ${pureBase64.length})`);

        try {
            const payload = {
                model: "gemma3:4b", 
                messages: [{
                    role: "user",
                    content: "Analyze the primary vehicle in this image. Identify its main color and verify the license plate text. Return ONLY as 'COLOR: [vehicle_color], PLATE: [plate_text]'. If a value cannot be determined, use 'UNKNOWN' for that value.",
                    images: [pureBase64] 
                }],
                stream: false
            };
            const logPayload = {...payload, images: ["<base64_image_data_omitted>"]};
            console.debug(`${DEBUG_PREFIX} Sending to Vision API. Endpoint: ${ollamaApiEndpoint}, Payload:`, JSON.stringify(logPayload, null, 2)); // Changed to ollamaApiEndpoint
            
            const actualPayload = {...payload, images: [pureBase64]};

            const response = await fetch(ollamaApiEndpoint, { // Changed to ollamaApiEndpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(actualPayload)
            });
            console.debug(`${DEBUG_PREFIX} Vision API response status: ${response.status}`);

            let aiRawResponse = "No response content.";
            if (response.ok) {
                const result = await response.json();
                console.debug(`${DEBUG_PREFIX} Vision API response JSON:`, result);
                if (result.message && result.message.content) {
                    aiRawResponse = result.message.content;
                    console.debug(`${DEBUG_PREFIX} Vision API raw response content: "${aiRawResponse}"`);
                    analysisElement.innerHTML = `<p class="ai-analysis-title">NURVEK AI RAW OUTPUT:</p><p class="ai-analysis-report">${aiRawResponse.replace(/\n/g, '<br>')}</p>`;
                    
                    let parsedColor = "UNKNOWN"; let parsedPlate = "UNKNOWN";
                    const colorMatch = aiRawResponse.match(/COLOR:\s*([^,]+)/i);
                    if (colorMatch && colorMatch[1]) parsedColor = colorMatch[1].trim();
                    const plateMatch = aiRawResponse.match(/PLATE:\s*([A-Z0-9]+)/i); 
                    if (plateMatch && plateMatch[1]) parsedPlate = plateMatch[1].trim();
                    console.debug(`${DEBUG_PREFIX} Parsed from Vision API: Color="${parsedColor}", Plate="${parsedPlate}"`);

                    if (colorElement) colorElement.textContent = parsedColor !== "UNKNOWN" ? parsedColor : "AI Color N/A";
                    if (plateTextSpan) {
                        const originalOcrText = plateTextSpan.dataset.ocrText || plateTextSpan.textContent; 
                        plateTextSpan.dataset.ocrText = originalOcrText; 
                        if (parsedPlate !== "UNKNOWN" && parsedPlate !== originalOcrText.split(" ")[0]) { 
                             plateTextSpan.innerHTML = `${originalOcrText.split(" ")[0]} <span class="ai-verified-tag">(AI: ${parsedPlate})</span>`;
                        } else if (parsedPlate !== "UNKNOWN" && parsedPlate === originalOcrText.split(" ")[0]) {
                             plateTextSpan.innerHTML = `${originalOcrText.split(" ")[0]} <span class="ai-verified-tag">(AI Confirmed)</span>`;
                        } else if (parsedPlate !== "UNKNOWN") { 
                            plateTextSpan.innerHTML = `${parsedPlate} <span class="ai-verified-tag">(AI Override)</span>`;
                        }
                    }
                } else {
                    analysisElement.innerHTML = `<p class="ai-analysis-status">VISION AI: NO ANALYSIS DATA RETURNED.</p>`;
                    console.warn(`${DEBUG_PREFIX} Vision API response missing content for event: ${eventTimestamp}`, result);
                }
            } else {
                analysisElement.innerHTML = `<p class="ai-analysis-status">VISION AI COMMS ERROR [${response.status}] FOR EVENT ${eventTimestamp.slice(-6)}</p>`;
                console.error(`${DEBUG_PREFIX} Vision API request failed for event: ${eventTimestamp}, Status: ${response.status}, Text: ${response.statusText}`);
            }
            
            if (visionAiLogContentElement) { 
                 visionAiLogContentElement.innerHTML = `<p class="ai-analysis-title">LAST VISION AI RAW OUTPUT (Event: ${eventTimestamp.slice(-6)}):</p><p class="ai-analysis-report">${aiRawResponse.replace(/\n/g, '<br>') || 'No raw output to display.'}</p>`;
                 console.debug(`${DEBUG_PREFIX} Updated Vision AI Log tab with raw output for event ${eventTimestamp.slice(-6)}.`);
            }
        } catch (error) {
            console.error(`${DEBUG_PREFIX} Error calling Vision API for event ${eventTimestamp}:`, error);
            analysisElement.innerHTML = `<p class="ai-analysis-status">VISION AI OFFLINE // CONNECTION FAILED FOR EVENT ${eventTimestamp.slice(-6)}</p>`;
            if (visionAiLogContentElement) { 
                visionAiLogContentElement.innerHTML = `<p class="ai-analysis-title">LAST VISION AI STATUS (Event: ${eventTimestamp.slice(-6)}):</p><p class="ai-analysis-report">CONNECTION FAILED: ${error.message}</p>`;
            }
        } finally {
            if (aiThinkingBubbleElement) aiThinkingBubbleElement.classList.remove('visible');
            isVisionApiBusy = false;
            console.debug(`${DEBUG_PREFIX} analyzeEventImageWithVisionAPI finished for event: ${eventTimestamp}. isVisionApiBusy set to false.`);
            setTimeout(processVisionQueue, 500); 
        }
    }

    function processVisionQueue() {
        // ... (same as before)
        if (!isVisionApiBusy && visionAnalysisQueue.length > 0) {
            isVisionApiBusy = true; 
            const nextTask = visionAnalysisQueue.shift(); 
            console.debug(`${DEBUG_PREFIX} Processing vision task from queue for event ${nextTask.eventTimestamp.slice(-6)}. Queue size now: ${visionAnalysisQueue.length}. Card ID: ${nextTask.cardId}`);
            const cardElement = document.getElementById(nextTask.cardId);
            if (cardElement) {
                 console.debug(`${DEBUG_PREFIX} Card element found for vision task. Calling analyzeEventImageWithVisionAPI.`);
                 analyzeEventImageWithVisionAPI(nextTask.base64ImageData, cardElement, nextTask.eventTimestamp);
            } else {
                console.warn(`${DEBUG_PREFIX} Could not find card element for vision task: ${nextTask.cardId}. Resetting busy flag.`);
                isVisionApiBusy = false; 
            }
        }
    }
    setInterval(processVisionQueue, visionQueueProcessInterval);

    function createEventCard(eventData) {
        // ... (same as before)
        console.debug(`${DEBUG_PREFIX} createEventCard called for event timestamp: ${eventData.timestamp_utc}, Track ID: ${eventData.vehicle_track_id}`);
        const card = document.createElement('article');
        card.classList.add('event-card');
        const timestampForId = eventData.timestamp_utc.replace(/[^a-zA-Z0-9]/g, "");
        card.id = `event-card-${timestampForId}`; 
        const analysisDivId = `analysis-${timestampForId}`;
        const plateTextSpanId = `plate-ocr-${timestampForId}`;
        const vehicleColorSpanId = `color-ai-${timestampForId}`;

        let displayTimestamp = new Date(eventData.timestamp_utc).toLocaleString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });

        let lpText = 'NO PLATE DETECTED'; let lpConfidenceText = '';
        if (eventData.attributes && eventData.attributes.license_plate && eventData.attributes.license_plate.detected) {
            lpText = eventData.attributes.license_plate.text || 'PLATE DETECTED // OCR FAIL';
            if (eventData.attributes.license_plate.lp_detection_confidence !== null) {
                lpConfidenceText = `(DET.CONF: ${(eventData.attributes.license_plate.lp_detection_confidence * 100).toFixed(0)}%)`;
            }
        }
        let lpImageHTML = ''; let vehicleImageHTML = ''; let imageForVisionAnalysis = null;
        if (eventData.attributes && eventData.attributes.vehicle_image_base64) {
            vehicleImageHTML = `<div class="image-container"><p>VEHICLE IMAGE</p><img src="data:image/jpeg;base64,${eventData.attributes.vehicle_image_base64.substring(0,30)}..." alt="Vehicle Image" class="snapshot-image"></div>`;
            imageForVisionAnalysis = eventData.attributes.vehicle_image_base64;
        }
        if (eventData.attributes && eventData.attributes.license_plate && eventData.attributes.license_plate.image_base64) {
            lpImageHTML = `<div class="image-container"><p>LP SCAN</p><img src="data:image/jpeg;base64,${eventData.attributes.license_plate.image_base64.substring(0,30)}..." alt="LP Scan" class="snapshot-image"></div>`;
            if (!imageForVisionAnalysis) imageForVisionAnalysis = eventData.attributes.license_plate.image_base64;
        }
        console.debug(`${DEBUG_PREFIX} Event card details: LP Text="${lpText}", Has Vehicle Image=${!!vehicleImageHTML}, Has LP Image=${!!lpImageHTML}`);

        card.innerHTML = `
            <h3>TRACK ID: ${eventData.vehicle_track_id} // CLASS: ${eventData.base_class.toUpperCase()} // CONF: ${(eventData.confidence_base_class * 100).toFixed(0)}%</h3>
            <div class="event-details">
                <p><strong>TIMESTAMP:</strong> ${displayTimestamp}</p>
                <p><strong>CAMERA:</strong> ${eventData.camera_id}</p>
                <p><strong>LICENSE PLATE (OCR):</strong> <span class="original-plate-text" id="${plateTextSpanId}" data-ocr-text="${lpText}">${lpText}</span> ${lpConfidenceText}</p>
                <p><strong>VEHICLE COLOR (AI):</strong> <span class="vehicle-color-ai" id="${vehicleColorSpanId}">Pending AI...</span></p>
            </div>
            <div class="event-images">
                ${vehicleImageHTML.replace(eventData.attributes?.vehicle_image_base64?.substring(0,30)+'...', eventData.attributes?.vehicle_image_base64 || '')}
                ${lpImageHTML.replace(eventData.attributes?.license_plate?.image_base64?.substring(0,30)+'...', eventData.attributes?.license_plate?.image_base64 || '')}
            </div>
            <div class="event-ai-analysis-container" id="${analysisDivId}">
                <p class="ai-analysis-placeholder">Awaiting AI secondary analysis...</p>
            </div>
        `;
        
        if (imageForVisionAnalysis) {
            console.debug(`${DEBUG_PREFIX} Adding event ${eventData.timestamp_utc.slice(-6)} to visionAnalysisQueue. Queue size will be: ${visionAnalysisQueue.length + 1}`);
            visionAnalysisQueue.push({
                base64ImageData: imageForVisionAnalysis,
                cardId: card.id, 
                eventTimestamp: eventData.timestamp_utc
            });
        }
        console.debug(`${DEBUG_PREFIX} createEventCard finished for event ${eventData.timestamp_utc.slice(-6)}.`);
        return card;
    }

    async function fetchAndDisplayEvents() {
        // ... (same as before)
        if (!eventLogElement) return;
        try {
            const response = await fetch(apiLogEndpoint);
            if (!response.ok) {
                console.error(`${DEBUG_PREFIX} API Error fetching events. Status: ${response.status}`);
                const ph = eventLogElement.querySelector('.placeholder-text');
                const msg = `SYSTEM OFFLINE // EVENT LOG COMMS INTERRUPTED [${response.status}]`;
                if (ph) ph.textContent = msg; else if (eventLogElement.innerHTML.trim() === '') eventLogElement.innerHTML = `<p class="placeholder-text">${msg}</p>`;
                return;
            }
            const events = await response.json();
            console.debug(`${DEBUG_PREFIX} Fetched events from API. Count: ${events.length}. Events data (first event if any):`, events.length > 0 ? events[0] : 'No events');
            
            if (!Array.isArray(events)) {
                console.error(`${DEBUG_PREFIX} Invalid data format from Event API:`, events);
                const ph = eventLogElement.querySelector('.placeholder-text');
                const msg = `EVENT DATA CORRUPTION // INVALID PAYLOAD`;
                if (ph) ph.textContent = msg; else if (eventLogElement.innerHTML.trim() === '') eventLogElement.innerHTML = `<p class="placeholder-text">${msg}</p>`;
                return;
            }
            const placeholder = eventLogElement.querySelector('.placeholder-text');
            if (events.length === 0 && displayedEventTimestamps.size === 0) {
                if (placeholder) placeholder.textContent = 'NO CONFIRMED TARGETS IN LOG... MONITORING GRID...';
                return;
            }
            if (placeholder && events.length > 0) {
                placeholder.remove();
                console.debug(`${DEBUG_PREFIX} Removed placeholder from event log.`);
            }
            
            let newEventsAdded = false;
            events.forEach(event => {
                if (!event || typeof event.timestamp_utc === 'undefined') {
                    console.warn(`${DEBUG_PREFIX} Skipping malformed event object:`, event); return;
                }
                if (!displayedEventTimestamps.has(event.timestamp_utc)) {
                    console.debug(`${DEBUG_PREFIX} New event to display: ${event.timestamp_utc}`);
                    const card = createEventCard(event); 
                    eventLogElement.prepend(card);
                    displayedEventTimestamps.add(event.timestamp_utc);
                    newEventsAdded = true; totalEventsDisplayed++;
                    if (typeof anime === 'function') {
                        anime({ targets: card, opacity: [0, 1], translateX: [20, 0], duration: 500, easing: 'easeOutQuad' });
                    } else { card.style.opacity = 1; card.style.transform = 'translateX(0px)'; }
                }
            });
            if (newEventsAdded) {
                console.debug(`${DEBUG_PREFIX} New events added. Total displayed: ${totalEventsDisplayed}`);
                if (logCountElement) logCountElement.textContent = `(${totalEventsDisplayed})`;
            }
        } catch (error) {
            console.error(`${DEBUG_PREFIX} Critical Fetch/Display Error (Events):`, error);
            const ph = eventLogElement.querySelector('.placeholder-text');
            const msg = `SYSTEM CRITICAL ERROR // EVENT LOG INTERFACE FAILURE: ${error.message}`;
            if (ph || eventLogElement.innerHTML.trim() === '') eventLogElement.innerHTML = `<p class="placeholder-text">${msg}</p>`;
        }
    }

    async function fetchAndUpdateLiveFeed() {
        // ... (same as before)
        if (!liveFeedImageElement || !liveFeedOverlayElement) return;
        try {
            const response = await fetch(apiLiveFrameGetUrl);
            if (response.ok) {
                const apiResponse = await response.json();
                if (apiResponse && apiResponse.frame_base64) {
                    const newFrameSrc = `data:image/jpeg;base64,${apiResponse.frame_base64}`;
                    if (liveFeedImageElement.src !== newFrameSrc) {
                        liveFeedImageElement.src = newFrameSrc;
                    }
                    liveFeedImageElement.style.display = 'block';
                    while (liveFeedOverlayElement.firstChild) {
                        liveFeedOverlayElement.removeChild(liveFeedOverlayElement.firstChild);
                    }
                    if (apiResponse.detections && apiResponse.detections.length > 0) {
                        const imgRect = liveFeedImageElement.getBoundingClientRect();
                        const displayedWidth = imgRect.width; const displayedHeight = imgRect.height;
                        const nativeFrameWidth = liveFeedImageElement.naturalWidth; const nativeFrameHeight = liveFeedImageElement.naturalHeight;
                        if (nativeFrameWidth > 0 && nativeFrameHeight > 0 && displayedWidth > 0 && displayedHeight > 0) {
                            const scaleX = displayedWidth / nativeFrameWidth; const scaleY = displayedHeight / nativeFrameHeight;
                            liveFeedOverlayElement.setAttribute('viewBox', `0 0 ${displayedWidth} ${displayedHeight}`);
                            apiResponse.detections.forEach(det => {
                                const [x1, y1, x2, y2] = det.box_coords;
                                const rectX = x1 * scaleX; const rectY = y1 * scaleY;
                                const rectWidth = (x2 - x1) * scaleX; const rectHeight = (y2 - y1) * scaleY;
                                const svgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                                svgRect.setAttribute('x', rectX); svgRect.setAttribute('y', rectY);
                                svgRect.setAttribute('width', rectWidth); svgRect.setAttribute('height', rectHeight);
                                svgRect.setAttribute('class', 'esp-box');
                                liveFeedOverlayElement.appendChild(svgRect);
                                const svgText = document.createElementNS("http://www.w3.org/2000/svg", "text");
                                svgText.setAttribute('x', rectX + 3); svgText.setAttribute('y', rectY + 10);
                                svgText.setAttribute('class', 'esp-text');
                                let textContent = `${det.class_name.toUpperCase()}`;
                                if(det.track_id) textContent = `ID:${det.track_id} ${textContent}`;
                                svgText.textContent = textContent;
                                liveFeedOverlayElement.appendChild(svgText);
                            });
                        }
                    }
                }
            }
        } catch (error) { console.error(`${DEBUG_PREFIX} Error fetching/processing live feed:`, error); }
    }

    if (eventLogElement) {
        console.debug(`${DEBUG_PREFIX} Initializing event data stream polling.`);
        fetchAndDisplayEvents();
        setInterval(fetchAndDisplayEvents, 2500);
    }
    if (liveFeedImageElement && liveFeedOverlayElement) {
        console.debug(`${DEBUG_PREFIX} Initializing live feed & ESP stream polling.`);
        liveFeedImageElement.onload = () => { 
            console.debug(`${DEBUG_PREFIX} Live feed image loaded, calling fetchAndUpdateLiveFeed.`);
            fetchAndUpdateLiveFeed(); 
        };
        if (liveFeedImageElement.complete && liveFeedImageElement.src) {
            console.debug(`${DEBUG_PREFIX} Live feed image already complete, calling fetchAndUpdateLiveFeed.`);
            fetchAndUpdateLiveFeed();
        }
        setInterval(fetchAndUpdateLiveFeed, 200);
    }
    console.log('[NVK_CONSOLE_JS] Nurvek Interface Initialization Complete.');
});