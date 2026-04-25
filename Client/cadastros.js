const form = document.getElementById("cadastroForm");
const nomeInput = document.getElementById("nome");
const fotoInput = document.getElementById("foto");
const preview = document.getElementById("preview");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const statusEl = document.getElementById("status");
const statusLabel = document.getElementById("statusLabel");
const fileLabel = document.getElementById("fileLabel");
const submitBtn = document.getElementById("submitBtn");
const clearBtn = document.getElementById("clearBtn");

const SERVER_PORT = 8000;

function buildApiUrl(path) {
    const isHttpPage = window.location.protocol === "http:" || window.location.protocol === "https:";
    if (isHttpPage && window.location.port !== "5500") {
        return path;
    }

    const protocol = isHttpPage ? window.location.protocol : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${SERVER_PORT}${path}`;
}

function setStatus(message, level) {
    statusEl.textContent = message;
    statusEl.classList.remove("status-ok", "status-warn", "status-error");

    if (level === "ok") {
        statusEl.classList.add("status-ok");
        statusLabel.textContent = "Cadastrado";
        return;
    }

    if (level === "error") {
        statusEl.classList.add("status-error");
        statusLabel.textContent = "Erro";
        return;
    }

    statusEl.classList.add("status-warn");
    statusLabel.textContent = "Pendente";
}

function resetPreview() {
    preview.removeAttribute("src");
    preview.classList.remove("is-visible");
    previewPlaceholder.hidden = false;
    fileLabel.textContent = "-";
}

function resetForm() {
    form.reset();
    resetPreview();
    setStatus("Preencha os dados para cadastrar um rosto.", "warn");
    nomeInput.focus();
}

function handleFileChange() {
    const file = fotoInput.files && fotoInput.files[0];
    if (!file) {
        resetPreview();
        return;
    }

    fileLabel.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.classList.add("is-visible");
    previewPlaceholder.hidden = true;
    setStatus("Foto pronta para envio.", "warn");
}

async function handleSubmit(event) {
    event.preventDefault();

    const nome = nomeInput.value.trim();
    const foto = fotoInput.files && fotoInput.files[0];

    if (!nome || !foto) {
        setStatus("Informe o nome e selecione uma foto.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("nome", nome);
    formData.append("foto", foto);

    submitBtn.disabled = true;
    clearBtn.disabled = true;
    setStatus("Enviando cadastro para o servidor...", "warn");

    try {
        const response = await fetch(buildApiUrl("/cadastro"), {
            method: "POST",
            body: formData,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.detail || "Nao foi possivel cadastrar esse rosto.");
        }

        setStatus(payload.mensagem || "Rosto cadastrado com sucesso.", "ok");
    } catch (err) {
        setStatus(err.message || "Falha ao cadastrar rosto.", "error");
    } finally {
        submitBtn.disabled = false;
        clearBtn.disabled = false;
    }
}

fotoInput.addEventListener("change", handleFileChange);
clearBtn.addEventListener("click", resetForm);
form.addEventListener("submit", handleSubmit);

resetPreview();
