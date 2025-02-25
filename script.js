document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const messageInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const fileInput = document.getElementById("file-input");

    const API_URL = "http://127.0.0.1:5004/chat";
    const UPLOAD_URL = "http://127.0.0.1:5001/upload";
    const ANALYSIS_URL = "http://127.0.0.1:5003/analyze";
    const REPORT_URL = "http://127.0.0.1:5005/report_summary";

    let isAnalyzing = false;

    function loadChatHistory() {
        const savedHistory = localStorage.getItem("chatHistory");
        if (savedHistory) {
            chatBox.innerHTML = savedHistory;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        messageInput.value = "";
    }
    

    function saveChatHistory() {
        localStorage.setItem("chatHistory", chatBox.innerHTML);
    }
    function sanitizeText(text) {
        return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.innerHTML = `<strong>${sender}:</strong> ${sanitizeText(message)}`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    
       
        localStorage.setItem("chatHistory", chatBox.innerHTML);
    }
    

    async function fetchReportSummary() {
        console.log("🔍 Fetching report summary..."); // ✅ Debugging log
        try {
            const response = await fetch(REPORT_URL);
            const data = await response.json();
            console.log("📜 Report Summary Response:", data); // ✅ Debugging log
    
            if (data.response) {
                appendMessage("📜 Report Summary", data.response);
            } else {
                appendMessage("AI", "⚠️ No summary available.");
            }
        } catch (error) {
            console.error("❌ Error fetching report summary:", error);
            appendMessage("AI", "❌ Error retrieving report summary.");
        }
    }
    
    
    

    async function sendMessage(event) {
        if (event) {
            event.preventDefault();
            event.stopPropagation(); 
        }
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;
        appendMessage("You", userMessage);
        messageInput.value = "";

        if (userMessage.toLowerCase() === "clear") {
            chatBox.innerHTML = "";
            localStorage.removeItem("chatHistory");
            localStorage.removeItem("uploadedFileName");
            return;
        }

        if (userMessage.toLowerCase() === "analyze") {
            const uploadedFileName = localStorage.getItem("uploadedFileName");
            if (!uploadedFileName) {
                appendMessage("AI", "❌ No file uploaded. Please upload a file first.");
                return;
            }
            if (isAnalyzing) {
                appendMessage("AI", "⏳ Analysis is already in progress...");
                return;
            }
        
            isAnalyzing = true;
            appendMessage("AI", "🔍 Analyzing file...");
        
            try {
                const response = await fetch(ANALYSIS_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ file_name: uploadedFileName })
                });
        
                if (!response.ok) {
                    console.error("❌ Error fetching forensic report:", response.status, response.statusText);
                    appendMessage("AI", `❌ Error analyzing file. Server responded with ${response.status}`);
                    isAnalyzing = false;
                    return;
                }
        
                appendMessage("AI", "✅ Analysis complete.");
                await fetchReportSummary();
            } catch (error) {
                console.error("❌ Error during analysis:", error);
                appendMessage("AI", "❌ An error occurred while analyzing the file.");
            } finally {
                isAnalyzing = false;
            }
        }
        
        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage })
            });

            const data = await response.json();
            appendMessage("AI", data.response);
        } catch (error) {
            console.error("AI response error:", error);
           
        }
    }

    sendBtn.addEventListener("click", (event) => sendMessage(event));
    messageInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage(event);
    });

    uploadBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async (event) => {
        event.preventDefault();
        if (fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        const uploadedFileName = file.name;
        localStorage.setItem("uploadedFileName", uploadedFileName);
        appendMessage("AI", `📤 Uploading file: '${uploadedFileName}'...`);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(UPLOAD_URL, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (data.error) {
                appendMessage("AI", `❌ Error: ${data.error}`);
            } else {
                appendMessage("AI", `✅ Your file '${uploadedFileName}' has been uploaded. Type 'analyze' to start forensic analysis.`);
            }
        } catch (error) {
            console.error("Upload failed:", error);
            appendMessage("AI", "❌ Upload failed.");
        }
    });

    loadChatHistory();
});
