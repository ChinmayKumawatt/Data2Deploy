(function () {
    const previewButton = document.getElementById("preview-button");
    const fileInput = document.getElementById("dataset-file");
    const targetSelect = document.getElementById("target-column");
    const previewPanel = document.getElementById("preview-panel");
    const previewEmpty = document.getElementById("preview-empty");
    const trainButton = document.getElementById("train-button");
    const featureList = document.getElementById("feature-list");
    const previewRows = document.getElementById("preview-rows");
    const previewColumns = document.getElementById("preview-columns");
    const previewTable = document.getElementById("preview-table");
    const autoPanel = document.getElementById("auto-panel");
    const manualPanel = document.getElementById("manual-panel");
    const modeCards = Array.from(document.querySelectorAll(".mode-card"));
    const trainForm = document.getElementById("train-form");
    const loadingOverlay = document.getElementById("loading-overlay");

    if (!previewButton || !fileInput) {
        return;
    }

    function currentMode() {
        const checked = document.querySelector('input[name="feature_mode"]:checked');
        return checked ? checked.value : "auto";
    }

    function syncModeUI() {
        const mode = currentMode();
        modeCards.forEach((card) => {
            card.classList.toggle("active", card.dataset.mode === mode);
        });
        autoPanel.classList.toggle("hidden", mode !== "auto");
        manualPanel.classList.toggle("hidden", mode !== "manual");
    }

    function populateTargetOptions(columnNames) {
        targetSelect.innerHTML = "";
        columnNames.forEach((columnName) => {
            const option = document.createElement("option");
            option.value = columnName;
            option.textContent = columnName;
            targetSelect.appendChild(option);
        });
        refreshFeatureOptions(columnNames);
    }

    function refreshFeatureOptions(columnNames) {
        const target = targetSelect.value;
        featureList.innerHTML = "";

        columnNames
            .filter((columnName) => columnName !== target)
            .forEach((columnName) => {
                const label = document.createElement("label");
                label.className = "feature-chip";

                const input = document.createElement("input");
                input.type = "checkbox";
                input.name = "selected_features";
                input.value = columnName;

                const text = document.createElement("span");
                text.textContent = columnName;

                label.appendChild(input);
                label.appendChild(text);
                featureList.appendChild(label);
            });
    }

    function renderPreviewTable(rows) {
        const tableHead = previewTable.querySelector("thead");
        const tableBody = previewTable.querySelector("tbody");
        tableHead.innerHTML = "";
        tableBody.innerHTML = "";

        if (!rows || rows.length === 0) {
            return;
        }

        const headers = Object.keys(rows[0]);
        const headRow = document.createElement("tr");
        headers.forEach((header) => {
            const th = document.createElement("th");
            th.textContent = header;
            headRow.appendChild(th);
        });
        tableHead.appendChild(headRow);

        rows.forEach((row) => {
            const tr = document.createElement("tr");
            headers.forEach((header) => {
                const td = document.createElement("td");
                td.textContent = row[header];
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    }

    async function previewDataset() {
        const file = fileInput.files[0];
        if (!file) {
            window.alert("Choose a CSV file before previewing.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        previewButton.disabled = true;
        previewButton.textContent = "Inspecting...";

        try {
            const response = await fetch("/preview", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();

            if (!response.ok) {
                throw new Error(payload.detail || "Preview failed.");
            }

            previewRows.textContent = payload.rows;
            previewColumns.textContent = payload.column_names.length;
            populateTargetOptions(payload.column_names);
            renderPreviewTable(payload.sample_rows);
            previewEmpty.classList.add("hidden");
            previewPanel.classList.remove("hidden");
            trainButton.disabled = false;
            targetSelect.dataset.columns = JSON.stringify(payload.column_names);
        } catch (error) {
            window.alert(error.message);
        } finally {
            previewButton.disabled = false;
            previewButton.textContent = "Preview Dataset";
        }
    }

    previewButton.addEventListener("click", previewDataset);
    targetSelect.addEventListener("change", () => {
        const columnNames = JSON.parse(targetSelect.dataset.columns || "[]");
        refreshFeatureOptions(columnNames);
    });

    modeCards.forEach((card) => {
        card.addEventListener("click", () => {
            const radio = card.querySelector('input[type="radio"]');
            radio.checked = true;
            syncModeUI();
        });
    });

    syncModeUI();

    trainForm.addEventListener("submit", () => {
        loadingOverlay.classList.remove("hidden");
        trainButton.disabled = true;
    });
})();
