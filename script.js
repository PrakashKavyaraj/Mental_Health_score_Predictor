(() => {
  "use strict";

  // Dynamic API base resolution: use current origin if served via FastAPI, otherwise default to localhost:8000
  let API_BASE = window.location.origin.includes("http") && !window.location.origin.includes("file:")
    ? window.location.origin
    : "http://localhost:8000";

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const printBtn = document.getElementById("print-btn");
  const errorRetryBtn = document.getElementById("error-retry-btn");
  const applySimBtn = document.getElementById("apply-sim-btn");

  const stateIdle = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult = document.getElementById("state-result");
  const stateError = document.getElementById("state-error");
  const resultPanel = document.querySelector(".result-panel");

  const apiStatusText = document.getElementById("api-status-text");
  const apiStatusBadge = document.getElementById("api-status-badge");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl = document.getElementById("score-band");
  const scoreContextEl = document.getElementById("score-context");
  const confidencePill = document.getElementById("confidence-pill");
  const userScoreMarker = document.getElementById("user-score-marker");
  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeNeedle = document.getElementById("gauge-needle");
  const errorCopyEl = document.getElementById("error-copy");

  const positiveListEl = document.getElementById("positive-factors-list");
  const negativeListEl = document.getElementById("negative-factors-list");
  const recListEl = document.getElementById("recommendations-list");
  const baseNoteEl = document.getElementById("base-note");

  // Simulator Elements
  const simScreen = document.getElementById("sim-screen");
  const simSleep = document.getElementById("sim-sleep");
  const simExercise = document.getElementById("sim-exercise");
  const simUnlocks = document.getElementById("sim-unlocks");

  const simScreenVal = document.getElementById("sim-screen-val");
  const simSleepVal = document.getElementById("sim-sleep-val");
  const simExerciseVal = document.getElementById("sim-exercise-val");
  const simUnlocksVal = document.getElementById("sim-unlocks-val");

  const simOrigScore = document.getElementById("sim-orig-score");
  const simNewScore = document.getElementById("sim-new-score");
  const simDeltaPill = document.getElementById("sim-delta-pill");
  const simSummary = document.getElementById("sim-summary");

  const GAUGE_ARC_LENGTH = 283; // approx pi * r(90)
  let lastBaselinePayload = null;
  let simDebounceTimer = null;

  // ---------------------------------------------------------
  // Check API health & version
  // ---------------------------------------------------------
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        apiStatusText.textContent = `Online (${data.model_name || "LightGBM"} • ${data.model_size_kb ? Math.round(data.model_size_kb) : 96} KB)`;
        apiStatusBadge.classList.add("online");
        return;
      }
    } catch (_) {}
    apiStatusText.textContent = "Engine Ready (Local / Cloud Fallback)";
  }
  checkApiHealth();

  // ---------------------------------------------------------
  // Draw tick marks on gauge (radius 90, inner 78)
  // ---------------------------------------------------------
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 88, rInner = 78;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI;
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // ---------------------------------------------------------
  // Segmented control (stress_level) wiring
  // ---------------------------------------------------------
  const segGroup = document.getElementById("stress_level_group");
  const stressHiddenInput = document.getElementById("stress_level");
  segGroup.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      segGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      stressHiddenInput.value = btn.dataset.value;
      clearFieldError(stressHiddenInput);
    });
  });

  function setStressValue(val) {
    stressHiddenInput.value = val;
    segGroup.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === val);
    });
  }

  // ---------------------------------------------------------
  // Field-level error helpers
  // ---------------------------------------------------------
  function fieldWrapper(input) {
    return input.closest(".field");
  }

  function setFieldError(input, message) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.add("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = message;
  }

  function clearFieldError(input) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.remove("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = "";
  }

  function clearAllErrors() {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    form.querySelectorAll(".error-msg").forEach((m) => (m.textContent = ""));
  }

  // ---------------------------------------------------------
  // Client-side validation mirroring Pydantic schema
  // ---------------------------------------------------------
  function validate(payload) {
    const errors = [];
    const numericChecks = [
      ["age", 10, 100],
      ["avg_daily_usage_hours", 0, 24],
      ["daily_unlocks", 0, 1000],
      ["study_hours", 0, 24],
      ["physical_activity_hours", 0, 24],
      ["sleep_hours_per_night", 0, 24],
    ];

    numericChecks.forEach(([key, min, max]) => {
      const input = document.getElementById(key);
      const val = payload[key];
      if (val === "" || val === null || Number.isNaN(val)) {
        errors.push([input, "This field is required."]);
      } else if (val < min || val > max) {
        errors.push([input, `Must be between ${min} and ${max}.`]);
      }
    });

    ["gender", "country", "academic_level", "most_used_platform", "purpose_of_use"].forEach((key) => {
      const input = document.getElementById(key);
      if (!payload[key] || String(payload[key]).trim() === "") {
        errors.push([input, "This field is required."]);
      }
    });

    if (!payload.stress_level) {
      errors.push([stressHiddenInput, "Pick a stress level."]);
    }

    return errors;
  }

  function collectPayload() {
    const fd = new FormData(form);
    return {
      age: fd.get("age") === "" ? NaN : parseInt(fd.get("age"), 10),
      gender: fd.get("gender") || "",
      country: (fd.get("country") || "Other").trim(),
      academic_level: fd.get("academic_level") || "",
      most_used_platform: fd.get("most_used_platform") || "",
      purpose_of_use: fd.get("purpose_of_use") || "",
      avg_daily_usage_hours: fd.get("avg_daily_usage_hours") === "" ? NaN : parseFloat(fd.get("avg_daily_usage_hours")),
      daily_unlocks: fd.get("daily_unlocks") === "" ? NaN : parseInt(fd.get("daily_unlocks"), 10),
      study_hours: fd.get("study_hours") === "" ? NaN : parseFloat(fd.get("study_hours")),
      physical_activity_hours: fd.get("physical_activity_hours") === "" ? NaN : parseFloat(fd.get("physical_activity_hours")),
      sleep_hours_per_night: fd.get("sleep_hours_per_night") === "" ? NaN : parseFloat(fd.get("sleep_hours_per_night")),
      stress_level: fd.get("stress_level") || "",
    };
  }

  function fillPayload(data, autoSubmit = false) {
    document.getElementById("age").value = data.age;
    document.getElementById("gender").value = data.gender;
    document.getElementById("country").value = data.country;
    document.getElementById("academic_level").value = data.academic_level;
    document.getElementById("most_used_platform").value = data.most_used_platform;
    document.getElementById("purpose_of_use").value = data.purpose_of_use;
    document.getElementById("avg_daily_usage_hours").value = data.avg_daily_usage_hours;
    document.getElementById("daily_unlocks").value = data.daily_unlocks;
    document.getElementById("study_hours").value = data.study_hours;
    document.getElementById("physical_activity_hours").value = data.physical_activity_hours;
    document.getElementById("sleep_hours_per_night").value = data.sleep_hours_per_night;
    setStressValue(data.stress_level);
    clearAllErrors();

    if (autoSubmit) {
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else {
        form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    }
  }

  // ---------------------------------------------------------
  // Demo Profiles
  // ---------------------------------------------------------
  document.getElementById("demo-balanced")?.addEventListener("click", () => {
    fillPayload({
      age: 22,
      gender: "Female",
      country: "USA",
      academic_level: "Graduate",
      most_used_platform: "YouTube",
      purpose_of_use: "Education",
      avg_daily_usage_hours: 2.2,
      daily_unlocks: 75,
      study_hours: 5.5,
      physical_activity_hours: 2.0,
      sleep_hours_per_night: 8.5,
      stress_level: "Low"
    }, true);
  });

  document.getElementById("demo-strained")?.addEventListener("click", () => {
    fillPayload({
      age: 20,
      gender: "Male",
      country: "India",
      academic_level: "Undergraduate",
      most_used_platform: "Instagram",
      purpose_of_use: "Entertainment",
      avg_daily_usage_hours: 7.8,
      daily_unlocks: 230,
      study_hours: 1.2,
      physical_activity_hours: 0.2,
      sleep_hours_per_night: 4.8,
      stress_level: "Very High"
    }, true);
  });

  // ---------------------------------------------------------
  // UI State Switching
  // ---------------------------------------------------------
  function showState(name) {
    [stateIdle, stateLoading, stateResult, stateError].forEach((el) => (el.hidden = true));
    ({ idle: stateIdle, loading: stateLoading, result: stateResult, error: stateError }[name]).hidden = false;
  }

  function setSubmitting(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    submitBtn.classList.toggle("loading", isSubmitting);
  }

  function updateGauge(score) {
    const clamped = Math.max(0, Math.min(10, score));

    // Update needle rotation (-90deg at score 0, 0deg at score 5, +90deg at score 10)
    const angle = -90 + (clamped / 10) * 180;
    if (gaugeNeedle) {
      gaugeNeedle.style.transform = `rotate(${angle.toFixed(1)}deg)`;
    }

    // Update gauge arc fill
    if (gaugeFill) {
      gaugeFill.style.transition = "none";
      gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
      requestAnimationFrame(() => {
        gaugeFill.style.transition = "stroke-dashoffset 0.8s cubic-bezier(0.22, 1, 0.36, 1)";
        const offset = GAUGE_ARC_LENGTH * (1 - clamped / 10);
        gaugeFill.style.strokeDashoffset = String(offset);
      });
    }

    // Update population cohort bar
    const percentilePos = Math.max(5, Math.min(95, (clamped / 10) * 100));
    if (userScoreMarker) {
      userScoreMarker.style.left = `${percentilePos}%`;
    }
  }

  function renderResult(data) {
    const score = data.predicted_score;
    scoreNumberEl.textContent = score.toFixed(2);
    scoreBandEl.textContent = `Signal: ${data.wellness_band}`;
    scoreContextEl.textContent = data.status_summary;

    if (data.confidence_interval) {
      confidencePill.textContent = `90% Conf: [${data.confidence_interval.lower.toFixed(2)} – ${data.confidence_interval.upper.toFixed(2)}]`;
    }

    if (baseNoteEl && data.base_population_score) {
      baseNoteEl.textContent = `Base: ${data.base_population_score.toFixed(2)} pts`;
    }

    updateGauge(score);

    // Render Positive Drivers (SHAP)
    positiveListEl.innerHTML = "";
    if (data.positive_drivers && data.positive_drivers.length > 0) {
      data.positive_drivers.forEach((d) => {
        const chip = document.createElement("div");
        chip.className = "factor-chip positive";
        chip.innerHTML = `<span>${d.factor}</span><span class="factor-impact">${d.impact}</span>`;
        positiveListEl.appendChild(chip);
      });
    } else {
      positiveListEl.innerHTML = `<div class="factor-chip"><span>No strong positive factors detected.</span></div>`;
    }

    // Render Negative Detractors (SHAP)
    negativeListEl.innerHTML = "";
    if (data.negative_detractors && data.negative_detractors.length > 0) {
      data.negative_detractors.forEach((d) => {
        const chip = document.createElement("div");
        chip.className = "factor-chip negative";
        chip.innerHTML = `<span>${d.factor}</span><span class="factor-impact">${d.impact}</span>`;
        negativeListEl.appendChild(chip);
      });
    } else {
      negativeListEl.innerHTML = `<div class="factor-chip positive"><span>Zero negative detractors! Exceptional balance.</span></div>`;
    }

    // Render Personalized Recommendations
    recListEl.innerHTML = "";
    if (data.actionable_recommendations) {
      data.actionable_recommendations.forEach((rec) => {
        const li = document.createElement("li");
        li.textContent = rec;
        recListEl.appendChild(li);
      });
    }

    // Initialize What-If Simulator with current values
    initSimulator(lastBaselinePayload, score);

    showState("result");

    // Smooth scroll down to the analytics panel
    if (resultPanel) {
      resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderError(copy) {
    errorCopyEl.textContent = copy || "Could not connect to model server.";
    showState("error");
  }

  // ---------------------------------------------------------
  // Simulator Controls
  // ---------------------------------------------------------
  function initSimulator(payload, origScore) {
    if (!payload) return;
    simScreen.value = payload.avg_daily_usage_hours;
    simSleep.value = payload.sleep_hours_per_night;
    simExercise.value = payload.physical_activity_hours;
    simUnlocks.value = payload.daily_unlocks;

    simScreenVal.textContent = payload.avg_daily_usage_hours.toFixed(1);
    simSleepVal.textContent = payload.sleep_hours_per_night.toFixed(1);
    simExerciseVal.textContent = payload.physical_activity_hours.toFixed(1);
    simUnlocksVal.textContent = payload.daily_unlocks;

    simOrigScore.textContent = origScore.toFixed(2);
    simNewScore.textContent = origScore.toFixed(2);
    simDeltaPill.textContent = "+0.00 pts";
    simDeltaPill.className = "delta-pill";
    simSummary.textContent = "Slide controls to project immediate lifestyle gains.";
  }

  async function triggerSimulation() {
    if (!lastBaselinePayload) return;

    const changes = {
      avg_daily_usage_hours: parseFloat(simScreen.value),
      sleep_hours_per_night: parseFloat(simSleep.value),
      physical_activity_hours: parseFloat(simExercise.value),
      daily_unlocks: parseInt(simUnlocks.value, 10),
    };

    try {
      const res = await fetch(`${API_BASE}/api/v1/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseline: lastBaselinePayload, changes }),
      });

      if (!res.ok) return;
      const data = await res.json();

      simNewScore.textContent = data.simulated_score.toFixed(2);
      const delta = data.score_delta;
      const sign = delta > 0 ? "+" : "";
      simDeltaPill.textContent = `${sign}${delta.toFixed(2)} pts`;
      simDeltaPill.className = `delta-pill ${delta < 0 ? "negative" : ""}`;
      simSummary.textContent = data.delta_summary;

      // Update gauge visually to preview simulated result
      updateGauge(data.simulated_score);
      scoreNumberEl.textContent = data.simulated_score.toFixed(2);
      scoreBandEl.textContent = `Signal: ${data.simulated_wellness_band} (Simulated)`;
    } catch (_) {}
  }

  [simScreen, simSleep, simExercise, simUnlocks].forEach((slider) => {
    slider.addEventListener("input", (e) => {
      if (e.target === simScreen) simScreenVal.textContent = parseFloat(e.target.value).toFixed(1);
      if (e.target === simSleep) simSleepVal.textContent = parseFloat(e.target.value).toFixed(1);
      if (e.target === simExercise) simExerciseVal.textContent = parseFloat(e.target.value).toFixed(1);
      if (e.target === simUnlocks) simUnlocksVal.textContent = e.target.value;

      clearTimeout(simDebounceTimer);
      simDebounceTimer = setTimeout(triggerSimulation, 150);
    });
  });

  // Apply simulated changes back to the main form
  applySimBtn?.addEventListener("click", () => {
    document.getElementById("avg_daily_usage_hours").value = simScreen.value;
    document.getElementById("sleep_hours_per_night").value = simSleep.value;
    document.getElementById("physical_activity_hours").value = simExercise.value;
    document.getElementById("daily_unlocks").value = simUnlocks.value;
    form.requestSubmit();
  });

  // ---------------------------------------------------------
  // Submit Handler
  // ---------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAllErrors();

    const payload = collectPayload();
    const clientErrors = validate(payload);

    if (clientErrors.length > 0) {
      clientErrors.forEach(([input, msg]) => input && setFieldError(input, msg));
      clientErrors[0][0]?.focus?.();
      return;
    }

    lastBaselinePayload = payload;
    setSubmitting(true);
    showState("loading");

    try {
      let res = await fetch(`${API_BASE}/api/v1/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`API returned ${res.status}`);
      }

      const data = await res.json();
      renderResult(data);
    } catch (err) {
      renderError("Could not connect to model server. Ensure FastAPI is running on localhost:8000.");
    } finally {
      setSubmitting(false);
    }
  });

  resetBtn?.addEventListener("click", () => {
    showState("idle");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  errorRetryBtn?.addEventListener("click", () => {
    showState("idle");
  });

  printBtn?.addEventListener("click", () => {
    window.print();
  });

})();
