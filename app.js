// =====================================================
// YOLOv8 Object Counter - FULL APP.JS
// =====================================================


// -------------------------
// ELEMENTOS DOM
// -------------------------

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const statusDiv = document.getElementById("status");
const fpsDiv = document.getElementById("fps");
const objectsDiv = document.getElementById("objects");

const cameraSelect = document.getElementById("cameraSelect");
const confidenceSlider = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const freezeButton = document.getElementById("freezeButton");


// -------------------------
// VARIABLES GLOBALES
// -------------------------

let session = null;
let stream = null;

let paused = false;
let confidence = 0.5;

let lastFrameTime = performance.now();

const MODEL_SIZE = 640;
const NUM_CLASSES = 80;
const NUM_BOXES = 8400;


// -------------------------
// PROVIDER (WebGPU / WASM)
// -------------------------

async function getProvider() {

    if ("gpu" in navigator) {
        statusDiv.innerHTML = "⚡ WebGPU disponible";
        return "webgpu";
    }

    statusDiv.innerHTML = "🧠 Usando WASM";
    return "wasm";
}


// -------------------------
// CARGAR MODELO YOLOv8
// -------------------------

async function loadModel() {

    try {

        statusDiv.innerHTML = "📦 Cargando YOLOv8...";

        const provider = await getProvider();

        session = await ort.InferenceSession.create(
            "models/yolov8n.onnx",
            {
                executionProviders: [provider]
            }
        );

        statusDiv.innerHTML = "✅ Modelo listo";

    } catch (err) {

        console.error(err);
        statusDiv.innerHTML = "❌ Error cargando modelo";
    }
}


// -------------------------
// CÁMARA
// -------------------------

async function loadCameras() {

    const devices = await navigator.mediaDevices.enumerateDevices();

    const cameras = devices.filter(d => d.kind === "videoinput");

    cameraSelect.innerHTML = "";

    cameras.forEach((cam, i) => {

        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.text = cam.label || `Cámara ${i + 1}`;

        cameraSelect.appendChild(opt);
    });
}


// -------------------------
// INICIAR CÁMARA
// -------------------------

async function startCamera(deviceId = null) {

    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }

    const constraints = {
        video: deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: "environment" },
        audio: false
    };

    stream = await navigator.mediaDevices.getUserMedia(constraints);

    video.srcObject = stream;

    await new Promise(res => {
        video.onloadedmetadata = () => {

            video.play();

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            res();
        };
    });

    statusDiv.innerHTML = "📷 Cámara lista";
}


// -------------------------
// EVENTOS UI
// -------------------------

cameraSelect.addEventListener("change", () => {
    startCamera(cameraSelect.value);
});

confidenceSlider.addEventListener("input", e => {
    confidence = parseFloat(e.target.value);
    confidenceValue.innerHTML = Math.round(confidence * 100) + "%";
});

freezeButton.addEventListener("click", () => {
    paused = !paused;
    freezeButton.innerHTML = paused ? "▶ Continuar" : "⏸ Congelar";
});


// -------------------------
// PREPROCESADO
// -------------------------

function preprocess() {

    const c = document.createElement("canvas");
    c.width = MODEL_SIZE;
    c.height = MODEL_SIZE;

    const cx = c.getContext("2d");

    cx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);

    const img = cx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);

    const data = img.data;

    const input = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);

    let r = 0;
    let g = MODEL_SIZE * MODEL_SIZE;
    let b = MODEL_SIZE * MODEL_SIZE * 2;

    for (let i = 0; i < data.length; i += 4) {

        input[r++] = data[i] / 255;
        input[g++] = data[i + 1] / 255;
        input[b++] = data[i + 2] / 255;
    }

    return new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}


// -------------------------
// IoU
// -------------------------

function iou(a, b) {

    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);

    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);

    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    const union = a.width * a.height + b.width * b.height - inter;

    return inter / union;
}


// -------------------------
// NMS POR CLASE
// -------------------------

function nmsByClass(boxes, threshold = 0.45) {

    const grouped = {};

    boxes.forEach(b => {

        if (!grouped[b.classId]) grouped[b.classId] = [];
        grouped[b.classId].push(b);
    });

    const result = [];

    for (const id in grouped) {

        let list = grouped[id];

        list.sort((a, b) => b.score - a.score);

        while (list.length) {

            const best = list.shift();

            result.push(best);

            list = list.filter(b => iou(best, b) < threshold);
        }
    }

    return result;
}


// -------------------------
// YOLO PARSER
// -------------------------

function parseYOLO(output) {

    const key = Object.keys(output)[0];
    const data = output[key].data;

    const boxes = [];

    for (let i = 0; i < NUM_BOXES; i++) {

        let max = 0;
        let cls = 0;

        for (let c = 0; c < NUM_CLASSES; c++) {

            const score = data[(c + 4) * NUM_BOXES + i];

            if (score > max) {
                max = score;
                cls = c;
            }
        }

        if (max > confidence) {

            const cx = data[0 * NUM_BOXES + i];
            const cy = data[1 * NUM_BOXES + i];
            const w = data[2 * NUM_BOXES + i];
            const h = data[3 * NUM_BOXES + i];

            boxes.push({
                x: cx - w / 2,
                y: cy - h / 2,
                width: w,
                height: h,
                score: max,
                classId: cls
            });
        }
    }

    return nmsByClass(boxes);
}


// -------------------------
// DIBUJO + CONTEO
// -------------------------

function drawObjects(objects) {

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const sx = canvas.width / MODEL_SIZE;
    const sy = canvas.height / MODEL_SIZE;

    const counter = {};

    objects.forEach(o => {

        const x = o.x * sx;
        const y = o.y * sy;
        const w = o.width * sx;
        const h = o.height * sy;

        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        const name = labels[o.classId];

        ctx.fillStyle = "#00FF00";
        ctx.font = "16px Arial";
        ctx.fillText(
            `${name} ${Math.round(o.score * 100)}%`,
            x,
            y > 20 ? y - 5 : y + 20
        );

        counter[name] = (counter[name] || 0) + 1;
    });

    objectsDiv.innerHTML =
        Object.keys(counter).length
            ? Object.entries(counter)
                .map(([k, v]) => `${k}: ${v}`)
                .join("<br>")
            : "No se detectan objetos";
}


// -------------------------
// LOOP PRINCIPAL
// -------------------------

async function detectLoop() {

    if (!paused && session) {

        const input = preprocess();

        const output = await session.run({
            images: input
        });

        const objects = parseYOLO(output);

        drawObjects(objects);

        const now = performance.now();

        const fps = Math.round(1000 / (now - lastFrameTime));

        lastFrameTime = now;

        fpsDiv.innerHTML = `FPS: ${fps}`;
    }

    requestAnimationFrame(detectLoop);
}


// -------------------------
// INICIO
// -------------------------

async function main() {

    await startCamera();
    await loadCameras();
    await loadModel();

    statusDiv.innerHTML = "🟢 Listo";

    detectLoop();
}

main();
