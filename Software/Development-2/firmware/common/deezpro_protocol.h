/**
 * =====================================================================
 * DEEZPRO FIRE EXTINGUISHER - SHARED PROTOCOL DEFINITIONS
 * =====================================================================
 * File: deezpro_protocol.h
 * Deskripsi: Definisi topik MQTT, string perintah, dan status baku
 *            untuk firmware ESP32.
 */

#ifndef DEEZPRO_PROTOCOL_H
#define DEEZPRO_PROTOCOL_H

// --- MQTT Topic Definitions ---
#define TOPIC_RAIL_X_CMD      "deezpro/rail/x/cmd"
#define TOPIC_RAIL_X_STATUS   "deezpro/rail/x/status"

#define TOPIC_RAIL_Y_CMD      "deezpro/rail/y/cmd"
#define TOPIC_RAIL_Y_STATUS   "deezpro/rail/y/status"

#define TOPIC_APAR_CMD        "deezpro/apar/cmd"
#define TOPIC_APAR_STATUS     "deezpro/apar/status"

#define TOPIC_SYSTEM_CMD      "deezpro/system/cmd"
#define TOPIC_SYSTEM_STATUS   "deezpro/system/status"

// --- Standard Status States ---
#define STATE_IDLE            "IDLE"
#define STATE_HOMING          "HOMING"
#define STATE_HOMED           "HOMED"
#define STATE_MOVING          "MOVING"
#define STATE_DONE            "DONE"
#define STATE_SPRAYING        "SPRAYING"
#define STATE_COMPLETED       "COMPLETED"
#define STATE_ERROR           "ERROR"

#endif // DEEZPRO_PROTOCOL_H
