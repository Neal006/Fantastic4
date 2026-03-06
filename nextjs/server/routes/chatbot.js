const express = require('express');
const { authenticate } = require('../middleware/auth');

const router = express.Router();
router.use(authenticate);

const mockResponses = {
    fault: [
        'Based on the SHAP analysis, the primary contributor to this fault is **elevated DC voltage**, which is {shapVal}% above normal range. This typically indicates a string voltage mismatch caused by one or more panels in the string producing higher voltage due to partial shading or module degradation.',
        'The fault on {inverterId} appears to be caused by **{faultType}**. The most significant factor is the DC current imbalance across strings. Recommend checking the string fusing and DC cable connections.',
        'Analysis shows **AC power output** has dropped significantly while DC input remains near normal — this is characteristic of an inverter trip or MPPT failure. Recommend a manual restart and monitoring for recurrence.',
    ],
    maintenance: [
        '**Maintenance Steps for {faultType}:**\n1. Isolate the inverter from the AC grid\n2. Check DC string fuses — replace any blown fuses\n3. Inspect DC cable connections for corrosion or loose contacts\n4. Verify string open-circuit voltages with a multimeter\n5. Re-energize and monitor for 30 minutes before resuming normal operation',
        '**Procedure:**\n- Shut down the inverter using the emergency stop\n- Wait 5 minutes for capacitors to discharge\n- Inspect all terminal connections for signs of arcing or overheating\n- Clean any dust accumulation from ventilation slots\n- Document findings and re-enable',
    ],
    general: [
        'Solar inverter performance degrades primarily due to: (1) **Thermal stress** — high temperatures reduce panel efficiency by ~0.5% per °C above 25°C; (2) **Soiling** — dust reduces output by 5-25% in arid regions; (3) **Shading** — partial shading has a disproportionate impact on MPPT efficiency.',
        'Category grades represent: **A (Healthy)** → All parameters normal; **B (Low Risk)** → Minor deviations, monitor closely; **C (Moderate)** → Attention needed, schedule inspection; **D (High Risk)** → Pre-fault condition, urgent inspection; **E (Critical)** → Active fault, immediate intervention required.',
        'SHAP values (SHapley Additive exPlanations) represent each feature\'s contribution to the AI model\'s fault prediction. Positive values push toward fault classification, negative values push toward healthy classification. The absolute magnitude indicates the strength of influence.',
    ],
};

function getRandomResponse(arr, replacements = {}) {
    let response = arr[Math.floor(Math.random() * arr.length)];
    for (const [key, value] of Object.entries(replacements)) {
        response = response.replace(new RegExp(`{${key}}`, 'g'), value);
    }
    return response;
}

// POST /api/chatbot/query
router.post('/query', async (req, res) => {
    try {
        const { message, context } = req.body;
        if (!message) return res.status(400).json({ error: 'Message is required' });

        const lowerMsg = message.toLowerCase();
        let category = 'general';
        if (lowerMsg.includes('fault') || lowerMsg.includes('causing') || lowerMsg.includes('why') || lowerMsg.includes('shap')) {
            category = 'fault';
        } else if (lowerMsg.includes('fix') || lowerMsg.includes('maintenance') || lowerMsg.includes('procedure') || lowerMsg.includes('how to')) {
            category = 'maintenance';
        }

        // Simulate typing delay
        await new Promise(r => setTimeout(r, 800 + Math.random() * 600));

        const replacements = {
            inverterId: context?.inverter_id || 'the inverter',
            faultType: context?.fault_type || 'the detected fault',
            shapVal: context?.shap_values ? Math.round(Math.max(...Object.values(context.shap_values)) * 100) : 42,
        };

        let responseText = getRandomResponse(mockResponses[category], replacements);

        // Add context-aware prefix if context provided
        if (context?.inverter_id) {
            responseText = `**Context: ${context.inverter_id}** (${context.current_category || 'unknown'} — ${context.fault_type || 'no active fault'})\n\n` + responseText;
        }

        res.json({
            response: responseText,
            context_used: !!context,
        });
    } catch (err) {
        console.error('Chatbot error:', err);
        res.status(500).json({ error: 'The AI assistant is currently unavailable. Please try again.' });
    }
});

module.exports = router;
