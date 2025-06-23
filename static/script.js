if (document.getElementById('upload')) {
    const uploadZone = document.getElementById('upload');
    const fileInput = document.getElementById('fileInput');
    const processing = document.getElementById('processing');
    const uploadedImage = document.getElementById('uploaded-image');
    const results = document.getElementById('results');
    const probability = document.getElementById('probability');
    const confidence = document.getElementById('confidence');
    const heatmapImg = document.getElementById('heatmap');
    const message = document.getElementById('message');
    const comparisonSection = document.getElementById('comparison-section');
    const compareUser = document.getElementById('compare-user');

    let uploadedImageDataURL = "";
    let heatmapImageDataURL = "";

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

            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImageDataURL = e.target.result;
                uploadedImage.src = uploadedImageDataURL;
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Prediction failed');

                const data = await response.json();
                if (data.error) throw new Error(data.error);

                setTimeout(() => {
                    processing.style.display = 'none';
                    results.style.display = 'block';
                    probability.textContent = `${data.predicted_class} (${Math.round(data.probability)}%)`;
                    confidence.textContent = `${Math.round(data.confidence)}%`;
                    heatmapImg.src = data.heatmap;
                    heatmapImg.style.display = 'block';
                    heatmapImageDataURL = data.heatmap;

                    const label = data.predicted_class.toLowerCase();

                    if (label === 'normal') {
                        message.className = 'message success';
                        message.innerHTML = `
                            🎉 <strong>Congratulations!</strong><br>Your scan shows <strong>no signs</strong> of lung cancer.<br><br>
                            <u>Precautions:</u> Maintain a healthy lifestyle, avoid smoking, eat a balanced diet, and schedule yearly health checkups.
                        `;
                        comparisonSection.style.display = 'none';
                    } else if (label === 'malignant') {
                        message.className = 'message warning';
                        message.innerHTML = `
                            ⚠️ <strong>Malignant condition detected!</strong><br>Please <strong>consult a medical specialist immediately.</strong><br><br>
                            <u>Recommendation:</u> Schedule a full medical diagnosis with your physician as soon as possible.
                        `;
                        comparisonSection.style.display = 'block';
                        compareUser.src = uploadedImageDataURL;
                    } else if (label === 'benign') {
                        message.className = 'message';
                        message.innerHTML = `
                            ✅ <strong>Benign condition detected.</strong><br>No immediate danger, but regular monitoring is advised.<br><br>
                            <u>Precautions:</u> Avoid pollution, monitor symptoms, and get checkups every 6–12 months.
                        `;
                        comparisonSection.style.display = 'none';
                    }
                }, 3000);
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
        message.innerHTML = '';
        comparisonSection.style.display = 'none';
    }

    const reportBtn = document.getElementById('saveReportBtn');
    const modal = document.getElementById('reportModal');
    const closeBtn = document.querySelector('.close');
    const downloadBtn = document.getElementById('downloadReport');

    reportBtn.addEventListener('click', () => modal.style.display = 'block');
    closeBtn.addEventListener('click', () => modal.style.display = 'none');
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });

    downloadBtn.addEventListener('click', () => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        const name = document.getElementById('userName').value;
        const email = document.getElementById('userEmail').value;
        const age = document.getElementById('userAge').value;
        const gender = document.getElementById('userGender').value;
        const contact = document.getElementById('userContact').value;
        const address = document.getElementById('userAddress').value;
        const comments = document.getElementById('userComments').value;
        const doctorEmail = document.getElementById('doctorEmail').value;
        const reportId = `RPT-${Date.now()}`;

        doc.setFontSize(16);
        doc.text('LungGuardian AI - Scan Report', 10, 10);

        doc.setFontSize(12);
        doc.text(`Report ID: ${reportId}`, 10, 20);
        doc.text(`Name: ${name}`, 10, 27);
        doc.text(`Email: ${email}`, 10, 34);
        doc.text(`Age: ${age}`, 10, 41);
        doc.text(`Gender: ${gender}`, 10, 48);
        doc.text(`Phone: ${contact}`, 10, 55);
        doc.text(`Address: ${address}`, 10, 62);
        doc.text(`Comments: ${comments}`, 10, 75);

        doc.setFontSize(13);
        doc.text(`Prediction: ${probability.textContent}`, 10, 85);
        doc.text(`Confidence: ${confidence.textContent}`, 10, 92);

        let summaryText = '';
        const label = probability.textContent.toLowerCase();
        if (label.includes('malignant')) {
            summaryText = 'Malignant signs detected. Urgent consultation with a medical specialist is recommended.';
        } else if (label.includes('benign')) {
            summaryText = 'Benign condition detected. Regular monitoring is advised.';
        } else {
            summaryText = 'No signs of lung cancer detected. Stay healthy and attend regular screenings.';
        }

        doc.setFontSize(11);
        doc.setFont("helvetica", "italic");
        doc.text('Summary:', 10, 102);
        doc.text(summaryText, 10, 110);

        let y = 120;

        // Page 1: Uploaded scan
        doc.setFont("helvetica", "normal");
        doc.text("Uploaded Scan:", 10, y);
        doc.addImage(uploadedImageDataURL, 'JPEG', 10, y + 5, 90, 60);
        y += 70;

        // Page 1: Heatmap
        doc.text("Result Heatmap:", 10, y);
        doc.addImage(heatmapImageDataURL, 'JPEG', 10, y + 5, 90, 60);

        // Page 2: Comparison
        const normalImage = new Image();
        normalImage.src = 'static/images/normal_case1.jpg'; // Ensure this path is correct
        normalImage.src = 'static/images/normal_case2.jpg'; // Ensure this path is correct
        normalImage.src = 'static/images/normal_case3.jpg'; // Ensure this path is correct

        normalImage.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = normalImage.width;
            canvas.height = normalImage.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(normalImage, 0, 0);
            const normalImageDataURL = canvas.toDataURL('image/jpg');

            doc.addPage();
            doc.setFontSize(14);
            doc.text('Comparison with Normal Scan', 10, 20);

            const imageWidth = 80;
            const imageHeight = 60;

            doc.setFontSize(12);
            doc.text("Your Scan", 25, 30);
            doc.text("Normal Scan", 120, 30);
            doc.addImage(uploadedImageDataURL, 'JPEG', 10, 35, imageWidth, imageHeight);
            doc.addImage(normalImageDataURL, 'JPEG', 110, 35, imageWidth, imageHeight);

            doc.save(`Scan_Report_${reportId}.pdf`);
            modal.style.display = 'none';
        };

        normalImage.onerror = () => {
            console.error('Error: Normal reference image not found');
            doc.addPage();
            doc.text("Normal reference image not found.", 10, 20);
            doc.save(`Scan_Report_${reportId}.pdf`);
            modal.style.display = 'none';
        };
    });
    
}
// --- Chatbot logic ---
const chatbotIcon = document.getElementById('chatbot-icon');
const chatbotBox = document.getElementById('chatbot-box');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotMessages = document.getElementById('chatbot-messages');
const chatbotClose = document.getElementById('chatbot-close');

chatbotIcon.onclick = () => chatbotBox.style.display = 'flex';
chatbotClose.onclick = () => chatbotBox.style.display = 'none';

chatbotInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && chatbotInput.value.trim()) {
        const userMsg = chatbotInput.value.trim();
        addChat('user', userMsg);
        respondToChat(userMsg.toLowerCase());
        chatbotInput.value = '';
    }
});

function addChat(sender, text) {
    const msg = document.createElement('div');
    msg.className = sender;
    msg.innerText = text;
    chatbotMessages.appendChild(msg);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

function respondToChat(input) {
    let response = "I'm not sure about that. Try asking about scan results, heatmaps, or when to consult a doctor.";

    if (input.includes('malignant')) {
        response = "A malignant result means potential cancer. You should consult a specialist immediately.";
    } else if (input.includes('benign')) {
        response = "Benign results are generally non-cancerous. It's best to monitor with regular follow-ups.";
    } else if (input.includes('normal')) {
        response = "Normal scans indicate no signs of lung cancer. Keep up healthy habits!";
    } else if (input.includes('heatmap')) {
        response = "The heatmap highlights areas where the AI detected possible abnormalities.";
    } else if (input.includes('confidence')) {
        response = "Confidence shows how sure the model is about its prediction. Higher is better.";
    } else if (input.includes('report')) {
        response = "Click 'Save as Report' to generate a PDF containing your scan and prediction.";
    } else if (input.includes('doctor')) {
        response = "Consult a doctor if your result is malignant, or if you have symptoms like coughing, chest pain, or shortness of breath.";
    } else if (input.includes('help')) {
        response = "Ask me about scan results, heatmaps, confidence scores, or when to consult a doctor.";
    }

    addChat('bot', response);
}
