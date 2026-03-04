import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [patient, setPatient] = useState({
    name: "",
    age: "",
    gender: "",
  });

  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Chat states
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);

  // Handle patient input
  const handlePatientChange = (e) => {
    setPatient({ ...patient, [e.target.name]: e.target.value });
  };

  // Handle image upload
  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  // Clear everything
  const clearAll = () => {
    setPatient({ name: "", age: "", gender: "" });
    setImage(null);
    setResult(null);
    setChatMessages([]);
    setLoading(false);
  };

  // Analyze image
  const analyzeImage = async () => {
    if (!patient.name.trim()) {
      alert("Please enter patient name");
      return;
    }

    if (!patient.age) {
      alert("Please enter valid age");
      return;
    }

    if (!patient.gender) {
      alert("Please select gender");
      return;
    }

    if (!image) {
      alert("Please upload a chest X-ray image");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/analyze",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(response.data);
    } catch (error) {
      alert("Error connecting to backend");
      console.error(error);
    }

    setLoading(false);
  };

  // Download report
  const downloadReport = async () => {
    if (!result) return;

    const payload = {
      name: patient.name,
      age: patient.age,
      gender: patient.gender,
      predicted_class: result.predicted_class,
      probabilities: result.probabilities,
      gradcam_image: result.gradcam_image,
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "medical_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();

      window.URL.revokeObjectURL(url);
    } catch (error) {
      alert("Error generating report");
      console.error(error);
    }
  };

  // Send chat message
  const sendMessage = async () => {
    if (!chatInput || !result) return;

    const userMessage = { sender: "user", text: chatInput };
    setChatMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease: result.predicted_class,
          question: chatInput,
        }),
      });

      const data = await response.json();

      const botMessage = { sender: "bot", text: data.response };
      setChatMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
    }

    setChatInput("");
  };

  return (
    <>
      {/* HEADER */}
      <div className="header">
        <h1>AI-Driven Lung Disease Detection</h1>
        <p>Deep Learning–based Chest X-ray Analysis</p>
      </div>

      <div className="container">
        <div className="grid">

          {/* Upload Card */}
          <div className="card">
            <div className="card-title">📤 Upload Chest X-ray</div>

            <label className="upload-box">
              Drag & drop X-ray image <br />
              <small>or click to browse</small>
              <input type="file" accept="image/*" onChange={handleImageChange} />
            </label>

            {image && (
              <p>
                <strong>Selected file:</strong> {image.name}
              </p>
            )}

            <div className="form-group">
              <label>Patient Name</label>
              <input
                type="text"
                name="name"
                value={patient.name}
                onChange={handlePatientChange}
              />
            </div>

            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                name="age"
                value={patient.age}
                onChange={(e) => {
                  const value = e.target.value;
                  if (/^\d*$/.test(value)) {
                    setPatient({ ...patient, age: value });
                  }
                }}
              />
            </div>

            <div className="form-group">
              <label>Gender</label>
              <select
                name="gender"
                value={patient.gender}
                onChange={handlePatientChange}
              >
                <option value="">Select</option>
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <button onClick={analyzeImage} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>

            {result && (
              <>
                <button
                  style={{ marginTop: "10px", background: "#6b7280" }}
                  onClick={clearAll}
                >
                  Clear / New Analysis
                </button>

                <button
                  style={{ marginTop: "10px", background: "#10b981" }}
                  onClick={downloadReport}
                >
                  Download Report
                </button>
              </>
            )}
          </div>

          {/* Prediction Card */}
          <div className="card">
            <div className="card-title">🧠 Prediction Result</div>

            {!result && <p>No Analysis Yet</p>}

            {result && (
              <>
                <h3>{result.predicted_class}</h3>
                <img
                  className="result-img"
                  src={`data:image/jpeg;base64,${result.gradcam_image}`}
                  alt="Grad-CAM"
                />
              </>
            )}
          </div>

          {/* Metrics Card */}
          <div className="card">
            <div className="card-title">📊 Model Metrics</div>

            {result &&
              Object.entries(result.probabilities).map(([key, value]) => (
                <div className="metric" key={key}>
                  <span>
                    {key} — {(value * 100).toFixed(1)}%
                  </span>
                  <div className="progress">
                    <div
                      style={{
                        width: `${value * 100}%`,
                        background: "#4f46e5",
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>

        </div>
      </div>

      {/* Floating Chat Icon */}
      <div className="chat-icon" onClick={() => setChatOpen(!chatOpen)}>
        💬
      </div>

      {/* Chat Window */}
      {chatOpen && (
        <div className="chat-window">
          <div className="chat-header">
            Medical Assistant
            <span
              style={{ cursor: "pointer" }}
              onClick={() => setChatOpen(false)}
            >
              ✖
            </span>
          </div>

          <div className="chat-body">
            {chatMessages.map((msg, index) => (
              <div
                key={index}
                className={msg.sender === "user" ? "chat-user" : "chat-bot"}
              >
                {msg.text}
              </div>
            ))}
          </div>

          <div className="chat-input">
            <input
              type="text"
              placeholder="Ask about the detected condition..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
