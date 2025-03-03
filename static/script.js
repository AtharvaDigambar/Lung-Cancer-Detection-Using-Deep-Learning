// Navigation (for index.html)
if (document.querySelector('nav')) {
    const links = document.querySelectorAll('nav a');
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            if (!link.href.includes('demo.html')) {
                e.preventDefault();
                document.querySelector(link.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// Detection functionality (for demo.html)
if (document.getElementById('upload')) {
    const uploadZone = document.getElementById('upload');
    const fileInput = document.getElementById('fileInput');
    const processing = document.getElementById('processing');
    const uploadedImage = document.getElementById('uploaded-image');
    const progress = document.getElementById('progress');
    const results = document.getElementById('results');
    const probability = document.getElementById('probability');
    const confidence = document.getElementById('confidence');
    const heatmapImg = document.getElementById('heatmap');
    const message = document.getElementById('message');

    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', (e) => e.preventDefault());
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    

    async function handleFile(file) {
        if (file && ['image/jpeg', 'image/png'].includes(file.type)) {
            uploadZone.style.display = 'none';
            processing.style.display = 'block';

            // Display uploaded image with overlay
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Prediction failed: ' + response.statusText);
                }

                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }

                setTimeout(() => {
                    processing.style.display = 'none';
                    results.style.display = 'block';
                    probability.textContent = `${data.predicted_class} (${Math.round(data.probability)}%)`;
                    confidence.textContent = `${Math.round(data.confidence)}%`;
                    heatmapImg.src = data.heatmap;
                    heatmapImg.style.display = 'block';

                    // Dynamic message based on prediction
                    if (data.predicted_class === 'normal') {
                        message.className = 'message success';
                        message.textContent = 'Congratulations! Your scan shows no signs of lung cancer.';
                    } else if (data.predicted_class === 'malignant') {
                        message.className = 'message warning';
                        message.textContent = 'Alert: Your scan indicates a malignant condition. Please consult a doctor immediately. Precautions: Quit smoking, avoid pollutants, and seek medical advice.';
                    } else if (data.predicted_class === 'benign') {
                        message.className = 'message';
                        message.textContent = 'Your scan shows a benign condition. Monitor with regular checkups.';
                    }
                }, 3000); // Simulated delay for loading
            } catch (error) {
                alert('Error: ' + error.message);
                reset();
            }
        } else {
            alert('Please upload a valid .jpg or .png file.');
        }
    }

    function reset() {
        results.style.display = 'none';
        uploadZone.style.display = 'block';
        uploadedImage.src = '';
        heatmapImg.style.display = 'none';
        message.textContent = '';
    }
}

// p5.js animation (optional, removed from demo page)