const canvas = document.getElementById('drawingBoard');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;


ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#FFFFFF';
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 25;


const outputCanvas = document.getElementById('outputBoard');
const outputCtx = outputCanvas.getContext('2d');


outputCtx.fillStyle = '#000000';
outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);


function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoordinates(e);
}

function stopDrawing() {
    isDrawing = false;
}

function draw(e) {
    if (!isDrawing) return;
    
    const [currentX, currentY] = getCoordinates(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    [lastX, lastY] = [currentX, currentY];
}

function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches && e.touches[0]) {
        return [
            (e.touches[0].clientX - rect.left) * scaleX,
            (e.touches[0].clientY - rect.top) * scaleY
        ];
    }
    
    return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY
    ];
}

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFFFFF';
}

async function generateMatrix() {
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/process_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
        
        const data = await response.json();
        console.log('Received data:', data);  
        
        if (!data.matrix) {
            throw new Error('No matrix data received');
        }
        
        
        displayMatrix(data.matrix);
        
        
        const predictionDiv = document.getElementById('prediction');
        if (data.prediction) {
            predictionDiv.textContent = `Predicted Letter: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
        }
    } catch (error) {
        console.error('Error:', error);
        const predictionDiv = document.getElementById('prediction');
        predictionDiv.textContent = `Error: ${error.message}`;
    }
}

function displayMatrix(matrix) {
    
    outputCtx.fillStyle = '#000000';
    outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    
    const pixelWidth = outputCanvas.width / 28;
    const pixelHeight = outputCanvas.height / 28;
    
    
    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
            const value = matrix[i][j];
            outputCtx.fillStyle = `rgb(${value},${value},${value})`;
            outputCtx.fillRect(
                j * pixelWidth,
                i * pixelHeight,
                pixelWidth,
                pixelHeight
            );
        }
    }
}


canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);


canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});
canvas.addEventListener('touchend', stopDrawing); 