import "./styles/styles.css";
import init, { MLP } from "mlp";
import { Chart, registerables } from "chart.js";
import { fill, range } from "es-toolkit";

async function fetchBinaryData(url: string) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

function setupPlot(): Chart {
    const plotCanvas = document.getElementById("plot")! as HTMLCanvasElement;
    Chart.register(...registerables);
    return new Chart(plotCanvas, {
        type: "bar",
        data: {
            labels: range(10).map((i) => i.toString()),
            datasets: [
                {
                    label: "Probability",
                    data: fill(Array(10), 0),
                    borderWidth: 1,
                },
            ],
        },
        options: {
            scales: {
                y: {
                    suggestedMin: 0,
                    suggestedMax: 1,
                    beginAtZero: true,
                },
            },
        },
    });
}

function setupCanvas(mlp: MLP, chart: Chart) {
    const canvas =
        document.querySelector<HTMLCanvasElement>("canvas#draw-area")!;
    const drawCtx = canvas.getContext("2d", { willReadFrequently: true })!;
    const resetButton = document.querySelector<HTMLButtonElement>("#reset")!;

    const scale = document.createElement("canvas");
    scale.width = scale.height = 28;
    const scaleCtx = scale.getContext("2d", { willReadFrequently: true })!;

    let isDown = false;

    function handleDown(evt: MouseEvent | TouchEvent) {
        evt.preventDefault();
        isDown = true;
        drawCtx.lineWidth = 20;
        drawCtx.lineJoin = "round";
        drawCtx.lineCap = "round";
        drawCtx.fillStyle = "black";
        drawCtx.beginPath();
    }

    function handleMouseMove(evt: MouseEvent) {
        var x = evt.offsetX;
        var y = evt.offsetY;
        if (isDown) {
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            doClassify();
        }
    }

    function handleTouchMove(evt: TouchEvent) {
        evt.preventDefault();
        var rect = canvas.getBoundingClientRect();
        var x = evt.touches[0].clientX - rect.left;
        var y = evt.touches[0].clientY - rect.top;
        if (isDown) {
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            doClassify();
        }
    }

    function handleUp() {
        isDown = false;
        drawCtx.stroke();
    }

    function reset() {
        drawCtx.fillStyle = "white";
        drawCtx.fillRect(0, 0, 280, 280);

        scaleCtx.fillStyle = "white";
        scaleCtx.fillRect(0, 0, 28, 28);

        chart.data.datasets[0].data = fill(Array(10), 0)
        chart.update()
    }

    function processImage() {
        const w = scaleCtx.canvas.width;
        const h = scaleCtx.canvas.height;

        scaleCtx.drawImage(canvas, 0, 0, w, h);

        const imageData = scaleCtx.getImageData(0, 0, w, h);
        scaleCtx.fillStyle = "white";
        scaleCtx.fillRect(0, 0, w, h);

        let minx = Infinity;
        let maxx = 0;
        let miny = Infinity;
        let maxy = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (imageData.data[4 * (y * w + x)] < 255) {
                    minx = Math.min(x - 5, minx);
                    maxx = Math.max(x + 5, maxx);
                    miny = Math.min(y - 5, miny);
                    maxy = Math.max(y + 5, maxy);
                }
            }
        }

        scaleCtx.drawImage(
            canvas,
            (canvas.width * minx) / w,
            (canvas.height * miny) / h,
            (canvas.width * (maxx - minx)) / w,
            (canvas.height * (maxy - miny)) / h,
            0,
            0,
            w,
            h,
        );
        return scaleCtx.getImageData(0, 0, w, h);
    }

    function doClassify() {
        const scaleData = processImage();

        const image = new Float32Array(28 * 28);
        image.forEach((_, i) => {
            image[i] = 1 - scaleData.data[i * 4] / 255;
        });

        const predictions = mlp.predict(image);
        chart.data.datasets[0].data = [...predictions];
        chart.update()

    }
    canvas.addEventListener("mousedown", handleDown);
    canvas.addEventListener("touchstart", handleDown);
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("touchmove", handleTouchMove);
    document.addEventListener("touchend", handleUp);
    document.addEventListener("mouseup", handleUp);
    resetButton.addEventListener("click", reset);
}

document.querySelector('#overlay')!.classList.remove('hidden');

init().then(async () => {
    const weights = await fetchBinaryData("/mnist_weights.bin");

    document.querySelector('#overlay')!.classList.add('hidden');

    const mlp = new MLP(weights);

    const chart = setupPlot();
    setupCanvas(mlp, chart);
});
