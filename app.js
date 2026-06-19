// ====================================
// VARIABLES GLOBALES
// ====================================

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


let session = null;
let stream = null;

let confidence = 0.5;
let paused = false;

let lastTime = performance.now();
let fps = 0;


// ====================================
// DETECTAR MOTOR DE EJECUCIÓN
// ====================================

async function getExecutionProvider() {

    if ("gpu" in navigator) {
        statusDiv.innerText = "WebGPU disponible";
        return "webgpu";
    }

    statusDiv.innerText = "Usando WASM";
    return "wasm";
}


// ====================================
// CARGAR MODELO YOLO
// ====================================

async function loadModel() {

    statusDiv.innerText = "Cargando YOLOv8...";

    const provider = await getExecutionProvider();

    try {

        session = await ort.InferenceSession.create(
            "models/yolov8n.onnx",
            {
                executionProviders: [provider]
            }
        );

        statusDiv.innerText =
            "Modelo cargado correctamente";

    } catch (error) {

        console.error(error);

        statusDiv.innerText =
            "Error cargando modelo";
    }
}


// ====================================
// LISTAR CÁMARAS
// ====================================

async function loadCameras() {

    const devices =
        await navigator.mediaDevices.enumerateDevices();

    const cameras =
        devices.filter(
            d => d.kind === "videoinput"
        );

    cameraSelect.innerHTML = "";

    cameras.forEach((cam, index) => {

        const option =
            document.createElement("option");

        option.value = cam.deviceId;

        option.text =
            cam.label || `Cámara ${index + 1}`;

        cameraSelect.appendChild(option);

    });
}


// ====================================
// INICIAR CÁMARA
// ====================================

async function startCamera(deviceId = null) {

    if (stream) {
        stream.getTracks()
              .forEach(t => t.stop());
    }


    const constraints = {

        video: deviceId
            ? {
                deviceId: {
                    exact: deviceId
                }
            }
            : {
                facingMode: "environment"
            },

        audio: false
    };


    stream =
        await navigator.mediaDevices
            .getUserMedia(constraints);


    video.srcObject = stream;


    await new Promise(resolve => {

        video.onloadedmetadata = () => {

            video.play();

            canvas.width =
                video.videoWidth;

            canvas.height =
                video.videoHeight;

            resolve();
        };

    });

}


// ====================================
// EVENTOS DE LA INTERFAZ
// ====================================


cameraSelect.addEventListener(
    "change",
    () => startCamera(
        cameraSelect.value
    )
);


confidenceSlider.addEventListener(
    "input",
    e => {

        confidence =
            parseFloat(e.target.value);

        confidenceValue.innerText =
            Math.round(confidence * 100) + "%";
    }
);


freezeButton.addEventListener(
    "click",
    () => {

        paused = !paused;

        freezeButton.innerText =
            paused
            ? "▶ Continuar"
            : "⏸ Congelar";

    }
);
// ====================================
// PREPARAR IMAGEN PARA YOLO
// Convierte la imagen a:
// [1, 3, 640, 640] Float32
// ====================================

function preprocess() {

    const size = 640;

    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");

    tempCanvas.width = size;
    tempCanvas.height = size;

    tempCtx.drawImage(
        video,
        0,
        0,
        size,
        size
    );

    const imageData =
        tempCtx.getImageData(
            0,
            0,
            size,
            size
        );

    const pixels = imageData.data;

    const input =
        new Float32Array(
            3 * size * size
        );

    let r = 0;
    let g = size * size;
    let b = 2 * size * size;


    // Convierte RGBA a RGB normalizado
    for (let i = 0; i < pixels.length; i += 4) {

        input[r++] = pixels[i] / 255;
        input[g++] = pixels[i + 1] / 255;
        input[b++] = pixels[i + 2] / 255;
    }


    return new ort.Tensor(
        "float32",
        input,
        [1, 3, size, size]
    );
}



// ====================================
// EJECUTAR YOLO
// ====================================

async function detect() {

    if (paused) {

        requestAnimationFrame(detect);
        return;
    }


    if (!session) {

        requestAnimationFrame(detect);
        return;
    }


    const start = performance.now();


    const inputTensor = preprocess();


    // Nombre del tensor de entrada.
    // En la mayoría de modelos YOLOv8
    // exportados es "images".
    const feeds = {
        images: inputTensor
    };


    const output =
        await session.run(feeds);


    const end = performance.now();


    fps = Math.round(
        1000 / (end - lastTime)
    );

    lastTime = end;


    fpsDiv.innerText =
        "FPS: " + fps;


    // En la próxima parte
    // vamos a interpretar
    // la salida del modelo

    requestAnimationFrame(detect);
}
// ====================================
// CALCULAR IoU (intersección sobre unión)
// Usado por NMS para eliminar duplicados
// ====================================

function iou(a, b) {

    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);

    const x2 = Math.min(
        a.x + a.width,
        b.x + b.width
    );

    const y2 = Math.min(
        a.y + a.height,
        b.y + b.height
    );

    const interArea =
        Math.max(0, x2 - x1) *
        Math.max(0, y2 - y1);

    const unionArea =
        a.width * a.height +
        b.width * b.height -
        interArea;

    return interArea / unionArea;
}


// ====================================
// NON MAXIMUM SUPPRESSION
// ====================================

function nms(boxes, threshold = 0.45) {

    boxes.sort((a, b) =>
        b.score - a.score
    );

    const result = [];

    while (boxes.length > 0) {

        const best = boxes.shift();

        result.push(best);

        boxes = boxes.filter(box =>
            iou(best, box) < threshold
        );
    }

    return result;
}
