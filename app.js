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
