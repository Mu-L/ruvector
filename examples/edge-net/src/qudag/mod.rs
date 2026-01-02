//! # QuDAG - Quantum-Resistant DAG Messaging
//!
//! WASM bindings for bitchat-qudag quantum-resistant messaging.
//! Provides ML-KEM-768 key encapsulation and ML-DSA signatures
//! for secure P2P communication in the edge-net.
//!
//! ## Features
//! - Post-quantum cryptography (ML-KEM-768, ML-DSA)
//! - DAG-based message ordering
//! - Multi-transport support (WebSocket, P2P, Bluetooth)
//! - Offline-first with sync on reconnect

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "qudag")]
use bitchat_qudag::{BitChatMessaging, Message as QuDagMessage, Transport};

/// Quantum-resistant messaging wrapper for WASM
#[wasm_bindgen]
pub struct WasmQuDag {
    node_id: String,
    #[cfg(feature = "qudag")]
    messenger: Option<BitChatMessaging>,
    /// Message DAG for ordering
    messages: Vec<QuDagMessageWrapper>,
    /// Connected peers
    peers: Vec<String>,
    /// Pending outbound messages
    pending: Vec<QuDagMessageWrapper>,
    /// Statistics
    stats: QuDagStats,
}

#[derive(Clone, Serialize, Deserialize)]
struct QuDagMessageWrapper {
    id: String,
    sender: String,
    recipient: Option<String>,
    content: Vec<u8>,
    timestamp: u64,
    parents: Vec<String>,
    signature: Vec<u8>,
    encrypted: bool,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct QuDagStats {
    messages_sent: u64,
    messages_received: u64,
    bytes_sent: u64,
    bytes_received: u64,
    peers_connected: usize,
    dag_depth: usize,
}

#[wasm_bindgen]
impl WasmQuDag {
    /// Create a new QuDAG messaging instance
    #[wasm_bindgen(constructor)]
    pub fn new(node_id: &str) -> WasmQuDag {
        WasmQuDag {
            node_id: node_id.to_string(),
            #[cfg(feature = "qudag")]
            messenger: None,
            messages: Vec::new(),
            peers: Vec::new(),
            pending: Vec::new(),
            stats: QuDagStats::default(),
        }
    }

    /// Initialize quantum-resistant messaging
    #[wasm_bindgen]
    pub fn init(&mut self) -> Result<bool, JsValue> {
        #[cfg(feature = "qudag")]
        {
            match BitChatMessaging::new(&self.node_id) {
                Ok(messenger) => {
                    self.messenger = Some(messenger);
                    Ok(true)
                }
                Err(e) => Err(JsValue::from_str(&format!("QuDAG init failed: {}", e))),
            }
        }

        #[cfg(not(feature = "qudag"))]
        {
            // Fallback mode without bitchat-qudag
            Ok(true)
        }
    }

    /// Get the node's public key (ML-KEM-768)
    #[wasm_bindgen(js_name = getPublicKey)]
    pub fn get_public_key(&self) -> Result<Vec<u8>, JsValue> {
        #[cfg(feature = "qudag")]
        if let Some(ref messenger) = self.messenger {
            return Ok(messenger.get_public_key().to_vec());
        }

        // Fallback: generate deterministic key from node_id
        let mut key = vec![0u8; 32];
        let id_bytes = self.node_id.as_bytes();
        for (i, byte) in id_bytes.iter().enumerate() {
            key[i % 32] ^= byte;
        }
        Ok(key)
    }

    /// Send an encrypted message to a peer
    #[wasm_bindgen(js_name = sendMessage)]
    pub fn send_message(
        &mut self,
        recipient: &str,
        content: &[u8],
    ) -> Result<String, JsValue> {
        let msg_id = format!("msg-{}-{}", self.node_id, self.stats.messages_sent);

        // Get parent messages (tips of DAG)
        let parents: Vec<String> = self.get_tips().into_iter().take(2).collect();

        #[cfg(feature = "qudag")]
        if let Some(ref mut messenger) = self.messenger {
            match messenger.send_encrypted(recipient, content) {
                Ok(signed_msg) => {
                    let wrapper = QuDagMessageWrapper {
                        id: msg_id.clone(),
                        sender: self.node_id.clone(),
                        recipient: Some(recipient.to_string()),
                        content: signed_msg.ciphertext.clone(),
                        timestamp: js_sys::Date::now() as u64,
                        parents,
                        signature: signed_msg.signature.clone(),
                        encrypted: true,
                    };
                    self.messages.push(wrapper);
                    self.stats.messages_sent += 1;
                    self.stats.bytes_sent += content.len() as u64;
                    return Ok(msg_id);
                }
                Err(e) => return Err(JsValue::from_str(&format!("Send failed: {}", e))),
            }
        }

        // Fallback mode
        let wrapper = QuDagMessageWrapper {
            id: msg_id.clone(),
            sender: self.node_id.clone(),
            recipient: Some(recipient.to_string()),
            content: content.to_vec(),
            timestamp: js_sys::Date::now() as u64,
            parents,
            signature: vec![0u8; 64], // Placeholder signature
            encrypted: false,
        };
        self.messages.push(wrapper.clone());
        self.pending.push(wrapper);
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += content.len() as u64;

        Ok(msg_id)
    }

    /// Broadcast a message to all peers
    #[wasm_bindgen(js_name = broadcastMessage)]
    pub fn broadcast_message(&mut self, content: &[u8]) -> Result<String, JsValue> {
        let msg_id = format!("bcast-{}-{}", self.node_id, self.stats.messages_sent);
        let parents: Vec<String> = self.get_tips().into_iter().take(2).collect();

        #[cfg(feature = "qudag")]
        if let Some(ref mut messenger) = self.messenger {
            match messenger.broadcast(content) {
                Ok(signed_msg) => {
                    let wrapper = QuDagMessageWrapper {
                        id: msg_id.clone(),
                        sender: self.node_id.clone(),
                        recipient: None,
                        content: content.to_vec(),
                        timestamp: js_sys::Date::now() as u64,
                        parents,
                        signature: signed_msg.signature.clone(),
                        encrypted: false,
                    };
                    self.messages.push(wrapper);
                    self.stats.messages_sent += 1;
                    self.stats.bytes_sent += content.len() as u64;
                    return Ok(msg_id);
                }
                Err(e) => return Err(JsValue::from_str(&format!("Broadcast failed: {}", e))),
            }
        }

        // Fallback mode
        let wrapper = QuDagMessageWrapper {
            id: msg_id.clone(),
            sender: self.node_id.clone(),
            recipient: None,
            content: content.to_vec(),
            timestamp: js_sys::Date::now() as u64,
            parents,
            signature: vec![0u8; 64],
            encrypted: false,
        };
        self.messages.push(wrapper.clone());
        self.pending.push(wrapper);
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += content.len() as u64;

        Ok(msg_id)
    }

    /// Receive and process an incoming message
    #[wasm_bindgen(js_name = receiveMessage)]
    pub fn receive_message(&mut self, data: &[u8]) -> Result<JsValue, JsValue> {
        #[cfg(feature = "qudag")]
        if let Some(ref mut messenger) = self.messenger {
            match messenger.receive(data) {
                Ok(msg) => {
                    self.stats.messages_received += 1;
                    self.stats.bytes_received += data.len() as u64;

                    // Add to DAG
                    let wrapper = QuDagMessageWrapper {
                        id: format!("recv-{}", self.stats.messages_received),
                        sender: msg.sender.clone(),
                        recipient: msg.recipient.clone(),
                        content: msg.content.clone(),
                        timestamp: msg.timestamp,
                        parents: msg.parents.clone(),
                        signature: msg.signature.clone(),
                        encrypted: msg.encrypted,
                    };
                    self.messages.push(wrapper);

                    return Ok(serde_wasm_bindgen::to_value(&msg)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?);
                }
                Err(e) => return Err(JsValue::from_str(&format!("Receive failed: {}", e))),
            }
        }

        // Fallback: deserialize as JSON
        match serde_json::from_slice::<QuDagMessageWrapper>(data) {
            Ok(msg) => {
                self.stats.messages_received += 1;
                self.stats.bytes_received += data.len() as u64;
                self.messages.push(msg.clone());
                Ok(serde_wasm_bindgen::to_value(&msg)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?)
            }
            Err(e) => Err(JsValue::from_str(&format!("Parse failed: {}", e))),
        }
    }

    /// Connect to a peer
    #[wasm_bindgen(js_name = connectPeer)]
    pub fn connect_peer(&mut self, peer_id: &str, public_key: &[u8]) -> Result<bool, JsValue> {
        #[cfg(feature = "qudag")]
        if let Some(ref mut messenger) = self.messenger {
            match messenger.add_peer(peer_id, public_key) {
                Ok(_) => {
                    self.peers.push(peer_id.to_string());
                    self.stats.peers_connected = self.peers.len();
                    return Ok(true);
                }
                Err(e) => return Err(JsValue::from_str(&format!("Connect failed: {}", e))),
            }
        }

        // Fallback mode
        if !self.peers.contains(&peer_id.to_string()) {
            self.peers.push(peer_id.to_string());
            self.stats.peers_connected = self.peers.len();
        }
        Ok(true)
    }

    /// Disconnect from a peer
    #[wasm_bindgen(js_name = disconnectPeer)]
    pub fn disconnect_peer(&mut self, peer_id: &str) -> bool {
        if let Some(pos) = self.peers.iter().position(|p| p == peer_id) {
            self.peers.remove(pos);
            self.stats.peers_connected = self.peers.len();

            #[cfg(feature = "qudag")]
            if let Some(ref mut messenger) = self.messenger {
                let _ = messenger.remove_peer(peer_id);
            }

            return true;
        }
        false
    }

    /// Get connected peers
    #[wasm_bindgen(js_name = getPeers)]
    pub fn get_peers(&self) -> Vec<String> {
        self.peers.clone()
    }

    /// Get DAG tips (messages with no children)
    #[wasm_bindgen(js_name = getTips)]
    pub fn get_tips(&self) -> Vec<String> {
        let all_parents: std::collections::HashSet<_> = self.messages
            .iter()
            .flat_map(|m| m.parents.iter())
            .collect();

        self.messages
            .iter()
            .filter(|m| !all_parents.contains(&m.id))
            .map(|m| m.id.clone())
            .collect()
    }

    /// Get message by ID
    #[wasm_bindgen(js_name = getMessage)]
    pub fn get_message(&self, msg_id: &str) -> Option<JsValue> {
        self.messages
            .iter()
            .find(|m| m.id == msg_id)
            .and_then(|m| serde_wasm_bindgen::to_value(m).ok())
    }

    /// Get recent messages (newest first)
    #[wasm_bindgen(js_name = getRecentMessages)]
    pub fn get_recent_messages(&self, count: usize) -> JsValue {
        let mut sorted: Vec<_> = self.messages.iter().collect();
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        let recent: Vec<_> = sorted.into_iter().take(count).collect();
        serde_wasm_bindgen::to_value(&recent).unwrap_or(JsValue::NULL)
    }

    /// Get pending outbound messages
    #[wasm_bindgen(js_name = getPendingMessages)]
    pub fn get_pending_messages(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.pending).unwrap_or(JsValue::NULL)
    }

    /// Clear pending messages after sync
    #[wasm_bindgen(js_name = clearPending)]
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Get DAG statistics
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> JsValue {
        let mut stats = self.stats.clone();
        stats.dag_depth = self.calculate_dag_depth();
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Calculate DAG depth
    fn calculate_dag_depth(&self) -> usize {
        if self.messages.is_empty() {
            return 0;
        }

        let mut depth_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for msg in &self.messages {
            let parent_depth = msg.parents
                .iter()
                .filter_map(|p| depth_map.get(p))
                .max()
                .unwrap_or(&0);
            depth_map.insert(msg.id.clone(), parent_depth + 1);
        }

        *depth_map.values().max().unwrap_or(&0)
    }

    /// Verify message signature
    #[wasm_bindgen(js_name = verifyMessage)]
    pub fn verify_message(&self, msg_id: &str) -> Result<bool, JsValue> {
        let msg = self.messages
            .iter()
            .find(|m| m.id == msg_id)
            .ok_or_else(|| JsValue::from_str("Message not found"))?;

        #[cfg(feature = "qudag")]
        if let Some(ref messenger) = self.messenger {
            return messenger.verify_signature(&msg.sender, &msg.content, &msg.signature)
                .map_err(|e| JsValue::from_str(&format!("Verify failed: {}", e)));
        }

        // Fallback: assume valid if signature present
        Ok(!msg.signature.is_empty())
    }

    /// Export DAG state for persistence
    #[wasm_bindgen(js_name = exportState)]
    pub fn export_state(&self) -> Result<Vec<u8>, JsValue> {
        serde_json::to_vec(&self.messages)
            .map_err(|e| JsValue::from_str(&format!("Export failed: {}", e)))
    }

    /// Import DAG state from persistence
    #[wasm_bindgen(js_name = importState)]
    pub fn import_state(&mut self, data: &[u8]) -> Result<usize, JsValue> {
        let messages: Vec<QuDagMessageWrapper> = serde_json::from_slice(data)
            .map_err(|e| JsValue::from_str(&format!("Import failed: {}", e)))?;

        let count = messages.len();
        self.messages = messages;
        self.stats.dag_depth = self.calculate_dag_depth();

        Ok(count)
    }

    /// Check if QuDAG has quantum-resistant crypto enabled
    #[wasm_bindgen(js_name = isQuantumResistant)]
    pub fn is_quantum_resistant(&self) -> bool {
        #[cfg(feature = "qudag")]
        return self.messenger.is_some();

        #[cfg(not(feature = "qudag"))]
        return false;
    }

    /// Get cryptographic algorithm info
    #[wasm_bindgen(js_name = getCryptoInfo)]
    pub fn get_crypto_info(&self) -> JsValue {
        #[cfg(feature = "qudag")]
        {
            let info = serde_json::json!({
                "keyEncapsulation": "ML-KEM-768",
                "digitalSignature": "ML-DSA-65",
                "hashFunction": "SHA3-256",
                "quantumResistant": self.messenger.is_some(),
                "keySize": 1184,
                "signatureSize": 3309
            });
            return serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL);
        }

        #[cfg(not(feature = "qudag"))]
        {
            let info = serde_json::json!({
                "keyEncapsulation": "fallback",
                "digitalSignature": "fallback",
                "hashFunction": "SHA-256",
                "quantumResistant": false,
                "keySize": 32,
                "signatureSize": 64
            });
            serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
        }
    }
}

/// Create a new QuDAG instance (convenience function)
#[wasm_bindgen(js_name = createQuDag)]
pub fn create_qudag(node_id: &str) -> Result<WasmQuDag, JsValue> {
    let mut qudag = WasmQuDag::new(node_id);
    qudag.init()?;
    Ok(qudag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qudag_creation() {
        let qudag = WasmQuDag::new("test-node");
        assert_eq!(qudag.node_id, "test-node");
        assert!(qudag.messages.is_empty());
    }

    #[test]
    fn test_dag_tips() {
        let mut qudag = WasmQuDag::new("test-node");

        // Add root message
        qudag.messages.push(QuDagMessageWrapper {
            id: "msg-1".to_string(),
            sender: "test-node".to_string(),
            recipient: None,
            content: vec![1, 2, 3],
            timestamp: 1000,
            parents: vec![],
            signature: vec![0; 64],
            encrypted: false,
        });

        // Add child message
        qudag.messages.push(QuDagMessageWrapper {
            id: "msg-2".to_string(),
            sender: "test-node".to_string(),
            recipient: None,
            content: vec![4, 5, 6],
            timestamp: 2000,
            parents: vec!["msg-1".to_string()],
            signature: vec![0; 64],
            encrypted: false,
        });

        let tips = qudag.get_tips();
        assert_eq!(tips.len(), 1);
        assert_eq!(tips[0], "msg-2");
    }
}
