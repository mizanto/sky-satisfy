// Function to validate the form data
function validateForm(formData) {
    const errors = {};
    // Check for presence and correctness of each field
    if (!formData.customer_type) errors.customer_type = 'Customer type is required.';
    if (!formData.age || formData.age < 0 || formData.age > 120) errors.age = 'Age must be between 0 and 120.';
    if (!formData.type_of_travel) errors.type_of_travel = 'Type of travel is required.';
    if (!formData.flight_distance) errors.flight_distance = 'Flight distance is required.';
    if (!formData.ease_of_online_booking || formData.ease_of_online_booking < 0 || formData.ease_of_online_booking > 5) {
        errors.ease_of_online_booking = 'Ease of online booking must be between 0 and 5.';
    }
    if (!formData.online_boarding || formData.online_boarding < 0 || formData.online_boarding > 5) {
        errors.online_boarding = 'Online boarding must be between 0 and 5.';
    }
    if (!formData.class) errors.class = 'Class is required.';
    return errors;
}

// Function to submit the prediction form
async function submitPredictionForm(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = ['age', 'flight_distance', 'ease_of_online_booking', 'online_boarding'].includes(key) ? parseInt(value, 10) : value;
    });

    const errors = validateForm(data);
    const resultDiv = document.getElementById('prediction-result');

    if (Object.keys(errors).length === 0) {
        resultDiv.innerHTML = '<p class="text-info">⏳ Waiting for prediction...</p>';
        resultDiv.style.display = 'block';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Server responded with an error');
            }

            const predictionData = await response.json();
            displayPredictionResult(predictionData, resultDiv);
        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerHTML = `<p class="alert alert-danger">Error: ${error.message}</p>`;
        }
    } else {
        displayValidationErrors(errors, resultDiv);
    }
}

// Function to fetch model information from the server
function fetchModelInfo() {
    fetch('/model/info')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            console.log('fetchModelInfo Response:', response);
            return response.json();
        })
        .then(data => {
            // Update UI with model information
            const modelInfoDiv = document.getElementById('model-info');
            modelInfoDiv.innerHTML = `
                <strong>Model Type:</strong> ${data.model_type}
                <p><strong>Training Date:</strong> ${data.training_date}</p>
                <strong>Metrics:</strong>
                <ul>
                    <li><strong>AUC:</strong> ${data.metrics.auc}</li>
                    <li><strong>F1:</strong> ${data.metrics.f1}</li>
                    <li><strong>Precision:</strong> ${data.metrics.precision}</li>
                    <li><strong>Recall:</strong> ${data.metrics.recall}</li>
                </ul>
            `;
            modelInfoDiv.style.display = 'block';
        })
        .catch(error => {
            // Handle any errors that occurred during the fetch
            console.error('Error:', error);
        });
}

function displayPredictionResult(predictionData, resultDiv) {
    const emoji = predictionData.verdict === 'satisfied' ? '🙂' : '😞';
    const verdictText = predictionData.verdict === 'satisfied' ? 'Happy Customer' : 'Sad Customer';
    const adviceText = predictionData.verdict === 'satisfied' ? 'Keep up the good work!' : 'You might want to look into this.';

    resultDiv.innerHTML = `
        <h2>${emoji}</h2>
        <h5>${verdictText}</h5>
        <p>${adviceText}</p>
    `;
}

function displayValidationErrors(errors, resultDiv) {
    const errorMessages = Object.values(errors).map(error => `<p class="alert alert-danger">${error}</p>`).join('');
    resultDiv.innerHTML = errorMessages;
    resultDiv.style.display = 'block';
}

// Set up event listeners when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set placeholder text for the prediction result area
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = '<p>🌟 Enter your flight details and unlock the power of prediction.</p>';

    // Attach the event listener to the form
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', submitPredictionForm);

    // Fetch and display model information
    console.log('Fetching model information...');
    fetchModelInfo();
});
