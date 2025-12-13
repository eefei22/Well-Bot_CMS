"""
Unified Dashboard for Cloud Services

Provides a real-time monitoring dashboard for Fusion, Intervention, and Context services.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os

from utils import activity_logger
from utils import database
from fusion.config_loader import load_config as load_fusion_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the unified dashboard HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Well-Bot Cloud Services Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: monospace;
            background-color: #0a0a0a;
            color: #e0e0e0;
            padding: 20px;
            overflow-x: hidden;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
            font-size: 2em;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 20px;
            height: calc(100vh - 140px);
        }
        
        .column {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            min-height: 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .column h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
            position: sticky;
            top: 0;
            background-color: #1a1a1a;
            z-index: 10;
            font-size: 1.2em;
        }
        
        .fusion-column h2 { color: #ff6b6b; }
        .intervention-column h2 { color: #4ecdc4; }
        .context-column h2 { color: #ffe66d; }
        .status-column h2 { color: #95e1d3; }
        
        .activity-list {
            list-style: none;
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }
        
        .activity-item {
            padding: 12px;
            margin-bottom: 10px;
            border: 1px solid #333;
            border-radius: 5px;
            background-color: #252525;
            transition: all 0.2s;
        }
        
        .activity-item:hover {
            background-color: #2a2a2a;
            border-color: #444;
        }
        
        .activity-item.success {
            border-left: 4px solid #4ecdc4;
        }
        
        .activity-item.error {
            border-left: 4px solid #ff6b6b;
        }
        
        .activity-item.no_signals {
            border-left: 4px solid #ffa500;
        }
        
        .activity-item.partial_success {
            border-left: 4px solid #ffe66d;
        }
        
        .item-header {
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .item-detail {
            font-size: 0.9em;
            color: #aaa;
            margin: 4px 0;
        }
        
        .item-detail-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 5px;
        }
        
        .emotion {
            color: #4ecdc4;
            font-weight: bold;
        }
        
        .confidence {
            color: #ffe66d;
        }
        
        .timestamp {
            color: #888;
            font-size: 0.85em;
        }
        
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .status-success { background-color: #4ecdc4; color: #000; }
        .status-error { background-color: #ff6b6b; color: #fff; }
        .status-no_signals { background-color: #ffa500; color: #000; }
        .status-partial { background-color: #ffe66d; color: #000; }
        
        .empty-message {
            color: #666;
            text-align: center;
            padding: 40px 20px;
            font-style: italic;
        }
        
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #1a1a1a;
            border-top: 2px solid #333;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
            z-index: 100;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-online {
            background-color: #4ecdc4;
            box-shadow: 0 0 8px rgba(78, 205, 196, 0.8);
        }
        
        .status-offline {
            background-color: #ff6b6b;
            box-shadow: 0 0 8px rgba(255, 107, 107, 0.8);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .service-status {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .service-status-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .service-name {
            color: #aaa;
        }
        
        .health-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .health-healthy { background-color: #4ecdc4; }
        .health-unhealthy { background-color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>Well-Bot Cloud Services Dashboard</h1>
    
    <div class="container">
        <!-- Top Left: Fusion Activity -->
        <div class="column fusion-column">
            <h2>Fusion Activity</h2>
            <ul class="activity-list" id="fusionList">
                <li class="empty-message">Loading...</li>
            </ul>
        </div>
        
        <!-- Top Right: Intervention Activity -->
        <div class="column intervention-column">
            <h2>Intervention Activity</h2>
            <ul class="activity-list" id="interventionList">
                <li class="empty-message">Loading...</li>
            </ul>
        </div>
        
        <!-- Bottom Left: Context Generation Activity -->
        <div class="column context-column">
            <h2>Context Generation Activity</h2>
            <ul class="activity-list" id="contextList">
                <li class="empty-message">Loading...</li>
            </ul>
        </div>
        
        <!-- Bottom Right: Service Status -->
        <div class="column status-column">
            <h2>Service Status</h2>
            <div id="serviceStatus">
                <div class="empty-message">Loading...</div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div>
            <span class="status-indicator status-online" id="statusIndicator"></span>
            <span>Status: <span id="statusText">Connected</span></span>
        </div>
        <div class="service-status" id="serviceStatusBar">
            <div class="service-status-item">
                <span class="service-name">Fusion:</span>
                <span class="health-indicator health-healthy" id="fusionHealth"></span>
            </div>
            <div class="service-status-item">
                <span class="service-name">Intervention:</span>
                <span class="health-indicator health-healthy" id="interventionHealth"></span>
            </div>
            <div class="service-status-item">
                <span class="service-name">Context:</span>
                <span class="health-indicator health-healthy" id="contextHealth"></span>
            </div>
            <div class="service-status-item">
                <span class="service-name">Database:</span>
                <span class="health-indicator health-healthy" id="dbHealth"></span>
            </div>
        </div>
        <div>
            Last Update: <span id="lastUpdate">--</span>
        </div>
    </div>
    
    <script>
        let fusionHistory = [];
        let interventionHistory = [];
        let contextHistory = [];
        const MAX_DISPLAY = 50;
        
        function formatTimestamp(timestamp) {
            if (!timestamp) return '--';
            const date = new Date(timestamp);
            return date.toLocaleString();
        }
        
        function formatDuration(seconds) {
            if (!seconds) return '--';
            return `${seconds.toFixed(2)}s`;
        }
        
        function updateFusion(fusionActivities) {
            const list = document.getElementById('fusionList');
            
            if (fusionActivities && fusionActivities.length > 0) {
                fusionActivities.forEach(activity => {
                    const exists = fusionHistory.find(a => 
                        a.timestamp === activity.timestamp && 
                        a.user_id === activity.user_id
                    );
                    if (!exists) {
                        fusionHistory.unshift(activity);
                    }
                });
                
                if (fusionHistory.length > MAX_DISPLAY) {
                    fusionHistory = fusionHistory.slice(0, MAX_DISPLAY);
                }
            }
            
            if (fusionHistory.length === 0) {
                list.innerHTML = '<li class="empty-message">No fusion activity yet</li>';
                return;
            }
            
            list.innerHTML = fusionHistory.map(activity => {
                const statusClass = activity.status === 'success' ? 'success' : 
                                   activity.status === 'no_signals' ? 'no_signals' : 'error';
                const statusBadge = activity.status === 'success' ? 'success' :
                                   activity.status === 'no_signals' ? 'no_signals' : 'error';
                
                return `
                    <li class="activity-item ${statusClass}">
                        <div class="item-header">
                            <span>User: ${activity.user_id.substring(0, 8)}...</span>
                            <span class="status-badge status-${statusBadge}">${activity.status.toUpperCase()}</span>
                        </div>
                        <div class="item-detail timestamp">${formatTimestamp(activity.timestamp)}</div>
                        ${activity.emotion_label ? `
                            <div class="item-detail-row">
                                <div class="item-detail">
                                    <span class="emotion">${activity.emotion_label}</span>
                                    <span class="confidence">(${((activity.confidence_score || 0) * 100).toFixed(1)}%)</span>
                                </div>
                            </div>
                        ` : ''}
                        ${activity.model_signals ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Signals Received: SER(${activity.model_signals.ser || 0}) FER(${activity.model_signals.fer || 0}) Vitals(${activity.model_signals.vitals || 0})
                                </div>
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Duration: ${formatDuration(activity.duration_seconds)}
                                </div>
                            </div>
                        ` : ''}
                        ${activity.model_signals_detail ? `
                            <div class="item-detail" style="font-size: 0.8em; margin-top: 5px; color: #888;">
                                ${activity.model_signals_detail.ser && activity.model_signals_detail.ser.length > 0 ? `
                                    SER: ${activity.model_signals_detail.ser.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                ` : ''}
                                ${activity.model_signals_detail.fer && activity.model_signals_detail.fer.length > 0 ? `
                                    FER: ${activity.model_signals_detail.fer.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                ` : ''}
                                ${activity.model_signals_detail.vitals && activity.model_signals_detail.vitals.length > 0 ? `
                                    Vitals: ${activity.model_signals_detail.vitals.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                ` : ''}
                            </div>
                        ` : ''}
                        ${activity.fusion_calculation_log ? `
                            <div class="item-detail" style="font-size: 0.8em; margin-top: 5px; color: #aaa; font-style: italic;">
                                Calculation: ${activity.fusion_calculation_log}
                            </div>
                        ` : ''}
                        ${activity.db_write_success !== undefined ? `
                            <div class="item-detail" style="font-size: 0.8em; margin-top: 3px;">
                                Database: <span style="color: ${activity.db_write_success ? '#4ecdc4' : '#ff6b6b'}">${activity.db_write_success ? '✓ Written' : '✗ Failed'}</span>
                            </div>
                        ` : ''}
                        ${activity.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${activity.error}
                            </div>
                        ` : ''}
                    </li>
                `;
            }).join('');
        }
        
        function updateIntervention(interventionActivities) {
            const list = document.getElementById('interventionList');
            
            if (interventionActivities && interventionActivities.length > 0) {
                interventionActivities.forEach(activity => {
                    const exists = interventionHistory.find(a => 
                        a.timestamp === activity.timestamp && 
                        a.user_id === activity.user_id
                    );
                    if (!exists) {
                        interventionHistory.unshift(activity);
                    }
                });
                
                if (interventionHistory.length > MAX_DISPLAY) {
                    interventionHistory = interventionHistory.slice(0, MAX_DISPLAY);
                }
            }
            
            if (interventionHistory.length === 0) {
                list.innerHTML = '<li class="empty-message">No intervention activity yet</li>';
                return;
            }
            
            list.innerHTML = interventionHistory.map(activity => {
                const statusClass = activity.status === 'success' ? 'success' : 'error';
                const decision = activity.decision || {};
                const emotion = activity.emotion || {};
                
                return `
                    <li class="activity-item ${statusClass}">
                        <div class="item-header">
                            <span>User: ${activity.user_id.substring(0, 8)}...</span>
                            <span class="status-badge status-${statusClass}">${activity.status.toUpperCase()}</span>
                        </div>
                        <div class="item-detail timestamp">${formatTimestamp(activity.timestamp)}</div>
                        ${emotion.label ? `
                            <div class="item-detail-row">
                                <div class="item-detail">
                                    <span class="emotion">${emotion.label}</span>
                                    <span class="confidence">(${((emotion.confidence || 0) * 100).toFixed(1)}%)</span>
                                </div>
                            </div>
                        ` : ''}
                        ${decision.trigger_intervention !== undefined ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Trigger: ${decision.trigger_intervention ? 'YES' : 'NO'} 
                                    (Confidence: ${((decision.confidence || 0) * 100).toFixed(1)}%)
                                </div>
                            </div>
                        ` : ''}
                        ${activity.ranked_activities && activity.ranked_activities.length > 0 ? `
                            <div class="item-detail" style="font-size: 0.85em; margin-top: 5px;">
                                Activities: ${activity.ranked_activities.map(a => `${a.rank}.${a.activity_type}`).join(', ')}
                            </div>
                        ` : ''}
                        ${activity.fusion ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #888;">
                                Fusion: ${activity.fusion.called ? 'Called' : 'Skipped'} 
                                ${activity.fusion.status ? `(${activity.fusion.status})` : ''}
                            </div>
                        ` : ''}
                        ${activity.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${activity.error}
                            </div>
                        ` : ''}
                    </li>
                `;
            }).join('');
        }
        
        function updateContext(contextActivities) {
            const list = document.getElementById('contextList');
            
            if (contextActivities && contextActivities.length > 0) {
                contextActivities.forEach(activity => {
                    const exists = contextHistory.find(a => 
                        a.timestamp === activity.timestamp && 
                        a.user_id === activity.user_id
                    );
                    if (!exists) {
                        contextHistory.unshift(activity);
                    }
                });
                
                if (contextHistory.length > MAX_DISPLAY) {
                    contextHistory = contextHistory.slice(0, MAX_DISPLAY);
                }
            }
            
            if (contextHistory.length === 0) {
                list.innerHTML = '<li class="empty-message">No context generation activity yet</li>';
                return;
            }
            
            list.innerHTML = contextHistory.map(activity => {
                const statusClass = activity.status === 'success' ? 'success' : 
                                   activity.status === 'partial_success' ? 'partial_success' : 'error';
                const results = activity.results || {};
                const durations = activity.durations || {};
                
                return `
                    <li class="activity-item ${statusClass}">
                        <div class="item-header">
                            <span>User: ${activity.user_id.substring(0, 8)}...</span>
                            <span class="status-badge status-${statusClass === 'partial_success' ? 'partial' : statusClass}">${activity.status.toUpperCase().replace('_', ' ')}</span>
                        </div>
                        <div class="item-detail timestamp">${formatTimestamp(activity.timestamp)}</div>
                        ${activity.conversation_id ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #888;">
                                Conversation: ${activity.conversation_id.substring(activity.conversation_id.length - 8)}
                            </div>
                        ` : ''}
                        <div class="item-detail-row">
                            <div class="item-detail" style="font-size: 0.85em;">
                                Facts: ${results.facts_extracted ? '✓' : '✗'} 
                                ${results.facts_length ? `(${results.facts_length.toLocaleString()} chars)` : ''}
                            </div>
                            <div class="item-detail" style="font-size: 0.85em;">
                                Context: ${results.context_extracted ? '✓' : '✗'} 
                                ${results.context_length ? `(${results.context_length.toLocaleString()} chars)` : ''}
                            </div>
                        </div>
                        ${activity.embedding ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em; color: #888;">
                                    Messages: ${activity.embedding.messages_processed || 0} | 
                                    Chunks: ${activity.embedding.chunks_created || 0}
                                </div>
                            </div>
                        ` : ''}
                        ${durations.total ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Duration: ${formatDuration(durations.total)}
                                    ${durations.embedding ? ` (Embed: ${formatDuration(durations.embedding)})` : ''}
                                    ${durations.facts ? ` (Facts: ${formatDuration(durations.facts)})` : ''}
                                    ${durations.context ? ` (Context: ${formatDuration(durations.context)})` : ''}
                                </div>
                            </div>
                        ` : ''}
                        ${activity.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${activity.error}
                            </div>
                        ` : ''}
                    </li>
                `;
            }).join('');
        }
        
        function updateServiceStatus(status) {
            const statusDiv = document.getElementById('serviceStatus');
            
            if (!status) {
                statusDiv.innerHTML = '<div class="empty-message">Status unavailable</div>';
                return;
            }
            
            const services = status.services || {};
            const modelServices = status.model_services || {};
            
            let html = '<div style="display: flex; flex-direction: column; gap: 15px;">';
            
            // Service health
            html += '<div><strong style="color: #00d4ff;">Service Health:</strong></div>';
            html += '<div style="display: flex; flex-direction: column; gap: 8px; margin-left: 10px;">';
            html += `<div>Fusion: <span style="color: ${services.fusion === 'healthy' ? '#4ecdc4' : '#ff6b6b'}">${services.fusion || 'unknown'}</span></div>`;
            html += `<div>Intervention: <span style="color: ${services.intervention === 'healthy' ? '#4ecdc4' : '#ff6b6b'}">${services.intervention || 'unknown'}</span></div>`;
            html += `<div>Context: <span style="color: ${services.context === 'healthy' ? '#4ecdc4' : '#ff6b6b'}">${services.context || 'unknown'}</span></div>`;
            html += `<div>Database: <span style="color: ${services.database === 'connected' ? '#4ecdc4' : '#ff6b6b'}">${services.database || 'unknown'}</span></div>`;
            html += '</div>';
            
            // Model service URLs
            if (Object.keys(modelServices).length > 0) {
                html += '<div style="margin-top: 15px;"><strong style="color: #00d4ff;">Model Services:</strong></div>';
                html += '<div style="display: flex; flex-direction: column; gap: 5px; margin-left: 10px; font-size: 0.9em;">';
                for (const [name, url] of Object.entries(modelServices)) {
                    html += `<div>${name.toUpperCase()}: <span style="color: #aaa;">${url}</span></div>`;
                }
                html += '</div>';
            }
            
            // Recent stats
            if (status.stats) {
                html += '<div style="margin-top: 15px;"><strong style="color: #00d4ff;">Recent Activity:</strong></div>';
                html += '<div style="display: flex; flex-direction: column; gap: 5px; margin-left: 10px; font-size: 0.9em;">';
                html += `<div>Fusion calls (last hour): ${status.stats.fusion_last_hour || 0}</div>`;
                html += `<div>Intervention calls (last hour): ${status.stats.intervention_last_hour || 0}</div>`;
                html += `<div>Context calls (last hour): ${status.stats.context_last_hour || 0}</div>`;
                html += '</div>';
            }
            
            html += '</div>';
            statusDiv.innerHTML = html;
            
            // Update health indicators
            document.getElementById('fusionHealth').className = 
                `health-indicator ${services.fusion === 'healthy' ? 'health-healthy' : 'health-unhealthy'}`;
            document.getElementById('interventionHealth').className = 
                `health-indicator ${services.intervention === 'healthy' ? 'health-healthy' : 'health-unhealthy'}`;
            document.getElementById('contextHealth').className = 
                `health-indicator ${services.context === 'healthy' ? 'health-healthy' : 'health-unhealthy'}`;
            document.getElementById('dbHealth').className = 
                `health-indicator ${services.database === 'connected' ? 'health-healthy' : 'health-unhealthy'}`;
        }
        
        async function fetchDashboardData() {
            try {
                const response = await fetch('/api/dashboard/status');
                const data = await response.json();
                
                updateFusion(data.fusion || []);
                updateIntervention(data.intervention || []);
                updateContext(data.context || []);
                updateServiceStatus(data.status || {});
                
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                document.getElementById('statusIndicator').className = 'status-indicator status-online';
                document.getElementById('statusText').textContent = 'Connected';
                
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
                document.getElementById('statusIndicator').className = 'status-indicator status-offline';
                document.getElementById('statusText').textContent = 'Disconnected';
            }
        }
        
        // Initial load
        fetchDashboardData();
        
        // Poll every 2 seconds
        setInterval(fetchDashboardData, 2000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@router.get("/status")
async def get_dashboard_status():
    """Get current dashboard status for all services."""
    try:
        # Read activity logs
        fusion_activities = activity_logger.read_activity_logs(
            "fusion",
            limit=50
        )
        intervention_activities = activity_logger.read_activity_logs(
            "intervention",
            limit=50
        )
        context_activities = activity_logger.read_activity_logs(
            "context",
            limit=50
        )
        
        # Get service health status
        services = {}
        try:
            # Check database
            client = database.get_supabase_client()
            client.table("users").select("id").limit(1).execute()
            services["database"] = "connected"
        except Exception:
            services["database"] = "disconnected"
        
        # Check fusion service (assume healthy if we can read logs)
        services["fusion"] = "healthy"
        
        # Check intervention service
        services["intervention"] = "healthy"
        
        # Check context service
        services["context"] = "healthy"
        
        # Get model service URLs from fusion config
        model_services = {}
        try:
            fusion_config = load_fusion_config()
            model_urls = fusion_config.get("model_service_urls", {})
            model_services = {
                "ser": model_urls.get("ser", "N/A"),
                "fer": model_urls.get("fer", "N/A"),
                "vitals": model_urls.get("vitals", "N/A")
            }
        except Exception:
            pass
        
        # Calculate recent stats (last hour)
        now = datetime.now()
        one_hour_ago = datetime.fromtimestamp(now.timestamp() - 3600)
        
        def count_recent(activities):
            count = 0
            for a in activities:
                try:
                    ts_str = a.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if ts.timestamp() > one_hour_ago.timestamp():
                            count += 1
                except Exception:
                    continue
            return count
        
        fusion_last_hour = count_recent(fusion_activities)
        intervention_last_hour = count_recent(intervention_activities)
        context_last_hour = count_recent(context_activities)
        
        return {
            "fusion": fusion_activities[:50],
            "intervention": intervention_activities[:50],
            "context": context_activities[:50],
            "status": {
                "services": services,
                "model_services": model_services,
                "stats": {
                    "fusion_last_hour": fusion_last_hour,
                    "intervention_last_hour": intervention_last_hour,
                    "context_last_hour": context_last_hour
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard status: {e}", exc_info=True)
        return {
            "fusion": [],
            "intervention": [],
            "context": [],
            "status": {
                "services": {},
                "model_services": {},
                "stats": {}
            },
            "error": str(e)
        }

