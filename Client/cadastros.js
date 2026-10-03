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
const thumbs = document.getElementById("thumbs");

const SERVER_PORT = 8000;
// O mesmo limite esta em MAX_FOTOS_CADASTRO, no Server/main.py.
const MAX_FOTOS = 5;
const FOTOS_SUGERIDAS = 3;
// Uma foto por vez: no celular, cada toque abre a camera e devolve uma foto.
const fotos = [];

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

function addFotos(files) {
    let ignoradas = 0;
    for (const file of files) {
        if (fotos.length >= MAX_FOTOS) {
            ignoradas += 1;
            continue;
        }
        fotos.push({ file, url: URL.createObjectURL(file) });
    }
    return ignoradas;
}

function removeFoto(index) {
    const [removida] = fotos.splice(index, 1);
    if (removida) {
        URL.revokeObjectURL(removida.url);
    }
}

function clearFotos() {
    fotos.splice(0).forEach(foto => URL.revokeObjectURL(foto.url));
}

function buildFormData(nome) {
    const formData = new FormData();
    formData.append("nome", nome);
    // Todas no mesmo campo: o servidor recebe a lista e guarda a media.
    fotos.forEach(foto => formData.append("foto", foto.file));
    return formData;
}

function fotosStatus() {
    if (fotos.length >= MAX_FOTOS) {
        return `${fotos.length} fotos prontas para envio, o maximo.`;
    }
    if (fotos.length >= FOTOS_SUGERIDAS) {
        return `${fotos.length} fotos prontas para envio. Cabem ate ${MAX_FOTOS}.`;
    }
    return `${fotos.length} de ${FOTOS_SUGERIDAS} fotos sugeridas. Com mais fotos, o reconhecimento fica mais firme.`;
}

function renderFotos() {
    thumbs.replaceChildren(...fotos.map((foto, index) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "thumb";
        item.title = `Remover foto ${index + 1}`;
        item.setAttribute("aria-label", `Remover foto ${index + 1}`);
        const img = document.createElement("img");
        img.src = foto.url;
        img.alt = "";
        item.append(img);
        item.addEventListener("click", () => {
            removeFoto(index);
            renderFotos();
        });
        return item;
    }));

    const ultima = fotos[fotos.length - 1];
    if (!ultima) {
        resetPreview();
        return;
    }
    fileLabel.textContent = `${fotos.length} de ${MAX_FOTOS}`;
    preview.src = ultima.url;
    preview.classList.add("is-visible");
    previewPlaceholder.hidden = true;
    setStatus(fotosStatus(), "warn");
}

function resetForm() {
    form.reset();
    clearFotos();
    renderFotos();
    setStatus("Preencha os dados para cadastrar um rosto.", "warn");
    nomeInput.focus();
}

function handleFileChange() {
    const ignoradas = addFotos(Array.from(fotoInput.files || []));
    // Limpa o campo para a proxima foto, mesmo que tenha o mesmo nome de arquivo.
    fotoInput.value = "";
    renderFotos();
    if (ignoradas > 0) {
        setStatus(`Cabem ${MAX_FOTOS} fotos; ${ignoradas} ficaram de fora.`, "warn");
    }
}

async function handleSubmit(event) {
    event.preventDefault();

    const nome = nomeInput.value.trim();

    if (!nome || fotos.length === 0) {
        setStatus("Informe o nome e adicione pelo menos uma foto.", "error");
        return;
    }

    const formData = buildFormData(nome);

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
