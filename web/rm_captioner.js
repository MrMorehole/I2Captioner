/**
 * RM Flux Captioner — web extension
 * Adds a "💾 Save as Preset" button to RMFluxCaptioner nodes.
 * Reads save_as_preset_name and custom_system_prompt widget values,
 * POSTs to /rm_captioner/save_preset, then refreshes the
 * style_preset dropdown live without restarting ComfyUI.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPES = new Set(["RMFluxCaptioner", "I2Caption"]);

// Cache of full preset data fetched from backend
let _presetCache = null;

async function fetchPresets() {
    if (_presetCache) return _presetCache;
    try {
        const resp = await fetch("/rm_captioner/get_presets");
        const data = await resp.json();
        if (data.status === "ok") {
            _presetCache = data.presets;
            return _presetCache;
        }
    } catch (e) {
        console.warn("[RM Captioner] Could not fetch presets:", e);
    }
    return null;
}

app.registerExtension({
    name: "RM.FluxCaptioner.SavePreset",

    async nodeCreated(node) {
        if (!NODE_TYPES.has(node.comfyClass)) return;
        addSaveButton(node);
        addPresetListener(node);
    },

    // Also handle nodes loaded from a saved workflow
    async loadedGraphNode(node) {
        if (!NODE_TYPES.has(node.comfyClass)) return;
        // Small delay to let widgets settle after load
        setTimeout(() => {
            addSaveButton(node);
            addPresetListener(node);
        }, 100);
    },
});


function addSaveButton(node) {
    // Avoid adding the button twice if nodeCreated and loadedGraphNode both fire
    if (node._rmSaveButtonAdded) return;
    node._rmSaveButtonAdded = true;

    // ComfyUI widget helpers
    const btn = node.addWidget("button", "💾 Save as Preset", null, () => {
        handleSave(node);
    }, { serialize: false });

    // Style the button to stand out
    btn.label = "💾 Save as Preset";
}


function getWidgetValue(node, name) {
    const w = node.widgets?.find(w => w.name === name);
    return w ? w.value : null;
}

function setWidgetOptions(node, name, options) {
    const w = node.widgets?.find(w => w.name === name);
    if (!w) return;
    w.options = w.options || {};
    w.options.values = options;
    // If current value is no longer in list, reset to first option
    if (!options.includes(w.value)) {
        w.value = options[0] ?? "";
    }
    node.setDirtyCanvas(true, true);
}


async function handleSave(node) {
    const presetName   = (getWidgetValue(node, "save_as_preset_name")        || "").trim();
    const customPrompt = (getWidgetValue(node, "custom_system_prompt") || "").trim();

    if (!presetName) {
        alert("Please enter a name in the 'save_as_preset_name' field before saving.");
        return;
    }
    if (!customPrompt) {
        alert("The 'custom_system_prompt' field is empty — nothing to save.");
        return;
    }

    const styleHint = (getWidgetValue(node, "style_hint") || "").trim();

    try {
        const resp = await fetch("/rm_captioner/save_preset", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ name: presetName, prompt: customPrompt, style_hint: styleHint }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            alert(`Save failed: ${data.error ?? resp.statusText}`);
            return;
        }

        // Refresh the style_preset dropdown with the updated list from server
        if (data.presets && Array.isArray(data.presets)) {
            setWidgetOptions(node, "style_preset", data.presets);

            // Select the newly saved preset so the user sees it immediately
            const styleWidget = node.widgets?.find(w => w.name === "style_preset");
            if (styleWidget) {
                styleWidget.value = presetName;
            }
        }

        // Clear the save_as_preset_name field as confirmation
        const nameWidget = node.widgets?.find(w => w.name === "save_as_preset_name");
        if (nameWidget) nameWidget.value = "";

        console.log(`[RM Captioner] Preset '${presetName}' saved successfully.`);

        // Brief visual feedback on the button
        const btn = node.widgets?.find(w => w.label === "💾 Save as Preset");
        if (btn) {
            const orig = btn.label;
            btn.label = "✅ Saved!";
            node.setDirtyCanvas(true, true);
            setTimeout(() => {
                btn.label = orig;
                node.setDirtyCanvas(true, true);
            }, 2000);
        }

    } catch (err) {
        alert(`Save error: ${err.message}`);
    }
}


// ---------------------------------------------------------------------------
// Preset dropdown → populate textareas
// ---------------------------------------------------------------------------

function addPresetListener(node) {
    if (node._rmPresetListenerAdded) return;
    node._rmPresetListenerAdded = true;

    const presetWidget = node.widgets?.find(w => w.name === "style_preset");
    if (!presetWidget) return;

    // Intercept value changes on the dropdown
    const origCallback = presetWidget.callback;
    presetWidget.callback = async function(value) {
        if (origCallback) origCallback.call(this, value);
        await loadPresetIntoWidgets(node, value);
    };
}

async function loadPresetIntoWidgets(node, presetName) {
    // Invalidate cache so a freshly saved preset is always picked up
    _presetCache = null;
    const presets = await fetchPresets();
    if (!presets) return;

    const preset = presets[presetName];
    if (!preset) return;

    // Populate custom_system_prompt
    const systemWidget = node.widgets?.find(w => w.name === "custom_system_prompt");
    if (systemWidget && preset.system_prompt) {
        systemWidget.value = preset.system_prompt;
    }

    // Populate style_hint if the preset carries one
    const hintWidget = node.widgets?.find(w => w.name === "style_hint");
    if (hintWidget && preset.style_hint) {
        hintWidget.value = preset.style_hint;
    } else if (hintWidget && !preset.style_hint) {
        // Clear the hint so stale text from the previous preset doesn't linger
        hintWidget.value = "";
    }

    node.setDirtyCanvas(true, true);
    console.log(`[RM Captioner] Loaded preset '${presetName}' into widgets.`);
}
