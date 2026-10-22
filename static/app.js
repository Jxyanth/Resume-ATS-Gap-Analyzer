let currentMode = "upload"; // "upload" | "text"
let selectedFile = null;
let lastAnalysisResult = null;

// Tab switcher between PDF upload and text paste
function switchResumeTab(mode) {
  currentMode = mode;
  const uploadZone = document.getElementById("uploadZone");
  const textZone = document.getElementById("textZone");
  const tabUploadBtn = document.getElementById("tabUploadBtn");
  const tabTextBtn = document.getElementById("tabTextBtn");

  if (mode === "upload") {
    uploadZone.classList.remove("hidden");
    textZone.classList.add("hidden");
    tabUploadBtn.classList.add("active");
    tabTextBtn.classList.remove("active");
  } else {
    uploadZone.classList.add("hidden");
    textZone.classList.remove("hidden");
    tabUploadBtn.classList.remove("active");
    tabTextBtn.classList.add("active");
  }
}

// Handle file selection
function handleFileSelected(input) {
  if (input.files && input.files[0]) {
    selectedFile = input.files[0];
    document.getElementById("dropFileName").innerHTML = `Selected: <strong>${escapeHtml(selectedFile.name)}</strong> (${(selectedFile.size / 1024).toFixed(1)} KB)`;
  }
}

// Drag & drop support
const dropZone = document.getElementById("uploadZone");
if (dropZone) {
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });
  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectedFile = e.dataTransfer.files[0];
      document.getElementById("resumeFileInput").files = e.dataTransfer.files;
      document.getElementById("dropFileName").innerHTML = `Selected: <strong>${escapeHtml(selectedFile.name)}</strong> (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    }
  });
}

// Sample data loader
function loadSampleData() {
  switchResumeTab("text");
  document.getElementById("resumeTextInput").value = 
`Jane Doe
Senior Machine Learning Engineer | San Francisco, CA | jane.doe@example.com

SUMMARY:
Machine Learning engineer with 6+ years of experience in deep learning, NLP, and distributed systems.
Proficient in Python, PyTorch, Docker, FastAPI, and Transformers.

EXPERIENCE:
Staff ML Engineer – AI Innovations (2021 – Present)
• Built scalable semantic search engine using PyTorch and FastAPI in Docker containers.
• Fine-tuned BERT and LLM models for token classification and sequence prediction.
• Maintained automated CI/CD pipelines with Git and Linux systems.

EDUCATION:
Bachelor of Science in Computer Science – University of California, Berkeley`;

  document.getElementById("jdTextInput").value = 
`Senior Machine Learning Engineer - Search & NLP

We are looking for a Senior Machine Learning Engineer to design and deploy state-of-the-art NLP models.

Requirements:
- 5+ years of software and machine learning engineering experience
- Strong proficiency in Python, PyTorch, Transformers, and HuggingFace
- Experience with Kubernetes, Docker, and FastAPI microservices
- Background in Large Language Models (LLM) and Sentence-Transformers
- Bachelor's or Master's degree in Computer Science or related quantitative field`;
}

// Run Analysis
async function runAnalysis() {
  const jdText = document.getElementById("jdTextInput").value.trim();
  if (!jdText) {
    alert("Please provide the target Job Description text.");
    return;
  }

  const formData = new FormData();
  formData.append("jd_text", jdText);

  if (currentMode === "upload") {
    if (!selectedFile) {
      alert("Please select or drop a PDF / text resume file.");
      return;
    }
    formData.append("resume_file", selectedFile);
  } else {
    const resumeText = document.getElementById("resumeTextInput").value.trim();
    if (!resumeText) {
      alert("Please paste your resume text.");
      return;
    }
    formData.append("resume_text", resumeText);
  }

  // Set loading UI
  setLoading(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Analysis request failed.");
    }

    const data = await response.json();
    lastAnalysisResult = data;
    renderResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const blankState = document.getElementById("blankState");
  const loadingState = document.getElementById("analysisLoading");
  const resultsContent = document.getElementById("resultsContent");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const spinner = document.getElementById("loadingSpinner");

  if (isLoading) {
    blankState.classList.add("hidden");
    resultsContent.classList.add("hidden");
    loadingState.classList.remove("hidden");
    analyzeBtn.disabled = true;
    spinner.classList.remove("hidden");
  } else {
    loadingState.classList.add("hidden");
    analyzeBtn.disabled = false;
    spinner.classList.add("hidden");
  }
}

function renderResults(data) {
  document.getElementById("blankState").classList.add("hidden");
  document.getElementById("analysisLoading").classList.add("hidden");
  document.getElementById("resultsContent").classList.remove("hidden");

  // Percentage and Dial
  const rawScore = data.overall_match_score || 0;
  const percentageVal = Math.round(rawScore * 100);
  const scoreText = document.getElementById("scoreText");
  const circlePath = document.getElementById("scoreCirclePath");
  const scoreTag = document.getElementById("scoreTag");

  scoreText.textContent = `${percentageVal}%`;
  circlePath.setAttribute("stroke-dasharray", `${percentageVal}, 100`);

  // Dial color & tag
  if (percentageVal >= 75) {
    circlePath.style.stroke = "var(--color-success)";
    scoreTag.textContent = "Strong Match";
    scoreTag.style.color = "var(--color-success)";
  } else if (percentageVal >= 55) {
    circlePath.style.stroke = "var(--color-warning)";
    scoreTag.textContent = "Moderate Fit";
    scoreTag.style.color = "var(--color-warning)";
  } else {
    circlePath.style.stroke = "var(--color-danger)";
    scoreTag.textContent = "Low Alignment";
    scoreTag.style.color = "var(--color-danger)";
  }

  // Count metrics
  const missingSkills = (data.missing_gaps && data.missing_gaps.missing_skills) || [];
  const matchedSkills = (data.matched_entities && data.matched_entities.matched_skills) || [];
  
  let resumeEntityCount = 0;
  if (data.resume_entities) {
    for (const k in data.resume_entities) resumeEntityCount += data.resume_entities[k].length;
  }
  let jdEntityCount = 0;
  if (data.jd_entities) {
    for (const k in data.jd_entities) jdEntityCount += data.jd_entities[k].length;
  }

  document.getElementById("metricMissingSkills").textContent = missingSkills.length;
  document.getElementById("metricMatchedSkills").textContent = matchedSkills.length;
  document.getElementById("metricResumeTotal").textContent = resumeEntityCount;
  document.getElementById("metricJdTotal").textContent = jdEntityCount;

  // Render Tag Clouds
  renderTagCloud("missingSkillsCloud", missingSkills, "missing", "No missing skills identified!");
  renderTagCloud("missingTitlesCloud", (data.missing_gaps && data.missing_gaps.missing_designations) || [], "missing", "No missing titles");
  renderTagCloud("missingDegreesCloud", (data.missing_gaps && data.missing_gaps.missing_degrees) || [], "missing", "No missing degree requirements");

  // Skills Breakdown
  const resSkills = (data.resume_entities && data.resume_entities.skills) || [];
  const jdSkills = (data.jd_entities && data.jd_entities.skills) || [];
  document.getElementById("countResumeSkills").textContent = resSkills.length;
  document.getElementById("countJdSkills").textContent = jdSkills.length;
  renderTagCloud("resumeSkillsCloud", resSkills, "neutral", "None detected");
  renderTagCloud("jdSkillsCloud", jdSkills, "neutral", "None detected");

  // Titles
  renderTagCloud("resumeTitlesCloud", (data.resume_entities && data.resume_entities.designations) || [], "neutral", "None detected");
  renderTagCloud("jdTitlesCloud", (data.jd_entities && data.jd_entities.designations) || [], "neutral", "None detected");

  // Degrees
  renderTagCloud("resumeDegreesCloud", (data.resume_entities && data.resume_entities.degrees) || [], "neutral", "None detected");
  renderTagCloud("jdDegreesCloud", (data.jd_entities && data.jd_entities.degrees) || [], "neutral", "None detected");

  // Raw JSON
  document.getElementById("rawJsonViewer").textContent = JSON.stringify(data, null, 2);
}

function renderTagCloud(containerId, items, type, emptyMsg) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";

  if (!items || items.length === 0) {
    const emptySpan = document.createElement("span");
    emptySpan.className = "badge-tag empty-notice";
    emptySpan.textContent = emptyMsg;
    container.appendChild(emptySpan);
    return;
  }

  items.forEach(item => {
    const span = document.createElement("span");
    span.className = `badge-tag ${type}`;
    span.textContent = item;
    container.appendChild(span);
  });
}

// Switch detail tabs
function showEntityCategory(tabName) {
  const tabs = document.querySelectorAll(".detail-tab");
  tabs.forEach(t => t.classList.remove("active"));
  event.target.classList.add("active");

  const paneMap = {
    gaps: "tabViewGaps",
    skills: "tabViewSkills",
    designations: "tabViewDesignations",
    degrees: "tabViewDegrees",
    rawJson: "tabViewRawJson",
  };

  for (const key in paneMap) {
    const el = document.getElementById(paneMap[key]);
    if (el) {
      if (key === tabName) {
        el.classList.remove("hidden");
      } else {
        el.classList.add("hidden");
      }
    }
  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, function(m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[m];
  });
}
