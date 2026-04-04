document.addEventListener('DOMContentLoaded', function() {
    
    // -- DOM ELEMENTS --
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileElem');
    const consoleLog = document.getElementById('console');
    const scanBtn = document.getElementById('scan-btn');
    const uploadText = document.getElementById('upload-text');
    const fileInfo = document.getElementById('file-info');
    
    // Safety Check: Only run this logic if we are on the Dashboard page
    if (!dropArea) return;

    // Store selected file
    let selectedFile = null;

    // -- EVENT LISTENERS --
    
    // Click to upload
    dropArea.addEventListener('click', () => fileInput.click());

    // Prevent default behaviors for drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover'); // Use CSS class for cleaner styling
        });
    });

    // Remove highlight when dragging leaves or drops
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        });
    });

    // Handle File Drop & Selection
    dropArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Scan button click
    scanBtn.addEventListener('click', startScan);

    // -- FUNCTIONS --

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            selectFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    }

    function selectFile(file) {
        selectedFile = file;
        uploadText.textContent = file.name;
        fileInfo.textContent = `${(file.size / 1024).toFixed(1)} KB - Click to change`;
        scanBtn.disabled = false;
        log(`File selected: ${file.name}`);
    }

    function startScan() {
        if (!selectedFile) return;
        scanBtn.disabled = true;
        scanBtn.textContent = 'Scanning...';
        processFile(selectedFile);
    }

    function log(message) {
        if (!consoleLog) return;
        const line = document.createElement('div');
        line.innerHTML = `> ${message}`;
        consoleLog.appendChild(line);
        consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    function updateStep(stepId, status) {
        const step = document.getElementById(stepId);
        if (!step) return;

        const statusText = step.querySelector('.status-text');
        
        // Reset classes
        step.classList.remove('waiting', 'active', 'completed');
        
        // Add new class based on status
        if (status === 'processing') {
            step.classList.add('active');
            statusText.textContent = 'Scanning...';
        } else if (status === 'complete') {
            step.classList.add('completed');
            statusText.textContent = 'Verified';
        } else if (status === 'flagged') {
            step.classList.add('completed'); // Keep green/red logic in CSS if needed
            statusText.textContent = 'Flagged';
        }
    }

    async function processFile(file) {
        // Update UI to show scanning
        uploadText.textContent = `Analyzing: ${file.name}...`;
        fileInfo.textContent = 'Processing...';

        log(`File received: ${file.name}`);
        log(`File type: ${file.type || 'Unknown'}`);
        log('Initiating 3-layer forensic analysis...');

        // Step 1: C2PA Check
        updateStep('step1', 'processing');
        log('[LAYER 1] Extracting C2PA provenance data...');

        // Prepare form data for upload
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Call backend API
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!result.success) {
                log(`❌ Error: ${result.error}`);
                scanBtn.disabled = false;
                scanBtn.textContent = 'Start Scan';
                fileInfo.textContent = 'Fix the issue and try again.';
                return;
            }

            // Update UI based on C2PA result
            const c2pa = result.layers.c2pa;
            if (c2pa && c2pa.c2pa_present) {
                updateStep('step1', 'complete');
                log(`[LAYER 1] C2PA metadata FOUND! Issuer: ${c2pa.issuer || 'Unknown'}`);
                log(`[LAYER 1] AI Generated Flag: ${c2pa.ai_generated ? 'YES' : 'NO'}`);
            } else {
                updateStep('step1', 'complete');
                log('[LAYER 1] No C2PA signature found. Proceeding to next layer...');
            }

            // Step 2: SynthID
            updateStep('step2', 'processing');
            log('[LAYER 2] SynthID check...');
            await delay(500);
            const synthid = result.layers.synthid;
            if (synthid?.status === 'complete') {
                if (synthid.is_watermarked) {
                    updateStep('step2', 'flagged');
                    log(`[LAYER 2] Watermark detected (${synthid.confidence.toFixed(1)}% confidence)`);
                } else {
                    updateStep('step2', 'complete');
                    log(`[LAYER 2] No watermark detected (${synthid.confidence.toFixed(1)}% confidence)`);
                }
            } else if (synthid?.status === 'skipped') {
                updateStep('step2', 'complete');
                log(`[LAYER 2] ${synthid.reason}`);
            } else if (synthid?.status === 'unavailable') {
                updateStep('step2', 'complete');
                log(`[LAYER 2] Unavailable - ${synthid.error || synthid.message || 'Detector unavailable'}`);
            } else if (synthid?.status === 'error') {
                updateStep('step2', 'complete');
                log(`[LAYER 2] Error - ${synthid.error || 'Detection failed'}`);
            } else {
                updateStep('step2', 'complete');
                log('[LAYER 2] No SynthID result returned.');
            }

            // Step 3: AI Model
            const aiModel = result.layers.ai_model;
            if (aiModel && aiModel.status === 'skipped') {
                updateStep('step3', 'complete');
                log(`[LAYER 3] Skipped - ${aiModel.reason}`);
            } else {
                updateStep('step3', 'processing');
                log('[LAYER 3] Running AI detection model...');
                await delay(500);

                if (aiModel && aiModel.status === 'complete') {
                    updateStep('step3', 'complete');
                    log(`[LAYER 3] Result: ${aiModel.label} (${aiModel.confidence.toFixed(1)}% confidence)`);
                } else {
                    log(`[LAYER 3] ${aiModel?.error || 'Model unavailable'}`);
                }
            }

            // Final verdict
            log('');
            log(`=== FINAL VERDICT: ${result.final_verdict} ===`);
            log(`Confidence: ${result.confidence.toFixed(1)}%`);

            // Store result in sessionStorage for report page
            sessionStorage.setItem('analysisResult', JSON.stringify(result));

            log('Generating forensic report...');
            await delay(1000);
            
            // Redirect to report page
            window.location.href = '/report';

        } catch (error) {
            log(`❌ Network error: ${error.message}`);
            console.error('Analysis error:', error);
            scanBtn.disabled = false;
            scanBtn.textContent = 'Start Scan';
            fileInfo.textContent = 'Connection problem. Please try again.';
        }
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
