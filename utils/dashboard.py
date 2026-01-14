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
import httpx

from utils import activity_logger
from utils import database
from fusion.config_loader import load_config as load_fusion_config
from utils.database import get_malaysia_timezone

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
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: 1fr;
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
        
        .model-services-column h2 { color: #ff6b6b; }
        .fusion-intervention-column h2 { color: #4ecdc4; }
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
        <!-- Left Column: Model Services Status -->
        <div class="column model-services-column">
            <h2>Model Services Status</h2>
            <ul class="activity-list" id="modelServicesList">
                <li class="empty-message">Loading...</li>
            </ul>
        </div>

        <!-- Middle Column: Fusion & Intervention Activity -->
        <div class="column fusion-intervention-column">
            <h2>Fusion & Intervention Activity</h2>
            <ul class="activity-list" id="fusionInterventionList">
                <li class="empty-message">Loading...</li>
            </ul>
        </div>

        <!-- Right Column: Service Status -->
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
        let modelServicesHistory = [];
        let fusionInterventionHistory = [];
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
        
        function updateModelServices(modelServicesData) {
            const list = document.getElementById('modelServicesList');

            if (!modelServicesData) {
                list.innerHTML = '<li class="empty-message">No model services data available</li>';
                return;
            }

            const services = [
                {
                    name: 'Vitals',
                    url: 'https://well-bot-bvs-emotion-520080168829.asia-south1.run.app',
                    key: 'vitals',
                    color: '#ff6b6b'
                },
                {
                    name: 'FER',
                    url: 'https://wellbot-fer-backend-520080168829.asia-southeast1.run.app',
                    key: 'fer',
                    color: '#4ecdc4'
                },
                {
                    name: 'SER',
                    url: 'https://well-bot-emotionrecognition-520080168829.asia-south1.run.app',
                    key: 'ser',
                    color: '#ffe66d'
                }
            ];

            list.innerHTML = services.map(service => {
                const serviceData = modelServicesData[service.key] || {};
                const isHealthy = serviceData.status === 'healthy' || serviceData.status === 'active';
                const lastActivity = serviceData.last_activity || {};
                const recentCount = serviceData.recent_signals || 0;

                            // Special handling for SER service with detailed status
                        if (service.key === 'ser') {
                            const queueSize = serviceData.queue_size || 0;
                            const currentProcessing = serviceData.current_processing;
                            const lastSuccessfulResult = serviceData.last_successful_result;

                            return `
                    <li class="activity-item ${isHealthy ? 'success' : 'error'}">
                        <div class="item-header">
                            <span style="color: ${service.color}">${service.name}</span>
                            <span class="status-badge status-${isHealthy ? 'success' : 'error'}">${isHealthy ? 'HEALTHY' : 'ERROR'}</span>
                        </div>
                        <div class="item-detail" style="font-size: 0.85em; color: #888;">
                            ${service.url}
                        </div>
                        <div class="item-detail-row">
                            <div class="item-detail" style="font-size: 0.85em;">
                                Queue Size: ${queueSize}
                            </div>
                            <div class="item-detail" style="font-size: 0.85em;">
                                Recent Signals: ${recentCount}
                            </div>
                        </div>
                        ${lastActivity.timestamp ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Last Request: ${formatTimestamp(lastActivity.timestamp)}
                                </div>
                                <div class="item-detail" style="font-size: 0.85em;">
                                    User: ${lastActivity.user_id ? lastActivity.user_id.substring(0, 8) + '...' : 'Unknown'}
                                </div>
                            </div>
                        ` : ''}
                        ${currentProcessing ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #ffe66d;">
                                Processing: ${currentProcessing.filename || 'No filename available'}
                            </div>
                            <div class="item-detail" style="font-size: 0.85em; color: #ffe66d;">
                                Started: ${formatTimestamp(currentProcessing.started_at)} (User: ${currentProcessing.user_id ? currentProcessing.user_id.substring(0, 8) + '...' : 'Unknown'})
                            </div>
                        ` : ''}
                        ${lastSuccessfulResult ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                Last Result: ${lastSuccessfulResult.emotion || 'N/A'} (${(lastSuccessfulResult.emotion_confidence * 100).toFixed(1)}%)
                            </div>
                            ${lastSuccessfulResult.sentiment ? `
                                <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                    Sentiment: ${lastSuccessfulResult.sentiment}
                                </div>
                            ` : ''}
                            ${lastSuccessfulResult.transcript ? `
                                <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                    Transcript: "${lastSuccessfulResult.transcript.length > 30 ? lastSuccessfulResult.transcript.substring(0, 30) + '...' : lastSuccessfulResult.transcript}"
                                </div>
                            ` : ''}
                            <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                Database write: ${lastSuccessfulResult.db_write_success ? 'processed (aggregation pending)' : 'failed'}
                            </div>
                        ` : ''}
                        ${serviceData.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${serviceData.error}
                            </div>
                        ` : ''}
                    </li>
                `;
                        } else if (service.key === 'fer') {
                            // Special handling for FER service with detailed status
                            const lastSuccessfulResult = serviceData.last_successful_result;

                            return `
                    <li class="activity-item ${isHealthy ? 'success' : 'error'}">
                        <div class="item-header">
                            <span style="color: ${service.color}">${service.name}</span>
                            <span class="status-badge status-${isHealthy ? 'success' : 'error'}">${isHealthy ? 'HEALTHY' : 'ERROR'}</span>
                        </div>
                        <div class="item-detail" style="font-size: 0.85em; color: #888;">
                            ${service.url}
                        </div>
                        <div class="item-detail-row">
                            <div class="item-detail" style="font-size: 0.85em;">
                                Recent Signals: ${recentCount}
                            </div>
                        </div>
                        ${lastActivity.timestamp ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Last Request: ${formatTimestamp(lastActivity.timestamp)}
                                </div>
                                <div class="item-detail" style="font-size: 0.85em;">
                                    User: ${lastActivity.user_id ? lastActivity.user_id.substring(0, 8) + '...' : 'Unknown'}
                                </div>
                            </div>
                        ` : ''}
                        ${lastSuccessfulResult ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                Last Result: ${lastSuccessfulResult.emotion || 'N/A'} (${(lastSuccessfulResult.emotion_confidence * 100).toFixed(1)}%)
                            </div>
                            <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                Aggregation: ${lastSuccessfulResult.aggregation_complete ? 'complete' : 'pending'}
                            </div>
                            <div class="item-detail" style="font-size: 0.85em; color: #4ecdc4;">
                                Database write: ${lastSuccessfulResult.db_write_success ? '✓ Written' : '✗ Failed'}
                            </div>
                        ` : ''}
                        ${serviceData.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${serviceData.error}
                            </div>
                        ` : ''}
                    </li>
                `;
                        } else {
                            // Default display for Vitals
                            return `
                    <li class="activity-item ${isHealthy ? 'success' : 'error'}">
                        <div class="item-header">
                            <span style="color: ${service.color}">${service.name}</span>
                            <span class="status-badge status-${isHealthy ? 'success' : 'error'}">${isHealthy ? 'HEALTHY' : 'ERROR'}</span>
                        </div>
                        <div class="item-detail" style="font-size: 0.85em; color: #888;">
                            ${service.url}
                        </div>
                        ${lastActivity.timestamp ? `
                            <div class="item-detail-row">
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Last Activity: ${formatTimestamp(lastActivity.timestamp)}
                                </div>
                                <div class="item-detail" style="font-size: 0.85em;">
                                    Recent Signals: ${recentCount}
                                </div>
                            </div>
                        ` : ''}
                        ${lastActivity.user_id ? `
                            <div class="item-detail" style="font-size: 0.85em; color: #aaa;">
                                Last User: ${lastActivity.user_id.substring(0, 8)}...
                            </div>
                        ` : ''}
                        ${serviceData.error ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em;">
                                Error: ${serviceData.error}
                            </div>
                        ` : ''}
                    </li>
                `;
                        }
            }).join('');
        }
        
        function updateFusionIntervention(activities) {
            const list = document.getElementById('fusionInterventionList');

            // Combine fusion and intervention activities, pairing by user_id and timestamp
            const combinedActivities = [];

            // Create a map of fusion activities by user_id + timestamp
            const fusionMap = new Map();
            if (activities.fusion && activities.fusion.length > 0) {
                activities.fusion.forEach(activity => {
                    const key = `${activity.user_id}_${activity.timestamp}`;
                    fusionMap.set(key, activity);
                });
            }

            // Process intervention activities and pair with fusion when available
            if (activities.intervention && activities.intervention.length > 0) {
                activities.intervention.forEach(intervention => {
                    const key = `${intervention.user_id}_${intervention.timestamp}`;
                    const fusion = fusionMap.get(key);

                    combinedActivities.push({
                        user_id: intervention.user_id,
                        timestamp: intervention.timestamp,
                        fusion: fusion,
                        intervention: intervention
                    });

                    // Remove from fusion map so we don't process unpaired fusions
                    if (fusion) {
                        fusionMap.delete(key);
                    }
                });
            }

            // Add remaining unpaired fusion activities
            fusionMap.forEach((fusion, key) => {
                combinedActivities.push({
                    user_id: fusion.user_id,
                    timestamp: fusion.timestamp,
                    fusion: fusion,
                    intervention: null
                });
            });

            // Sort by timestamp (newest first)
            combinedActivities.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

            if (combinedActivities.length > 0) {
                combinedActivities.forEach(activity => {
                    const exists = fusionInterventionHistory.find(a =>
                        a.timestamp === activity.timestamp &&
                        a.user_id === activity.user_id
                    );
                    if (!exists) {
                        fusionInterventionHistory.unshift(activity);
                    }
                });

                if (fusionInterventionHistory.length > MAX_DISPLAY) {
                    fusionInterventionHistory = fusionInterventionHistory.slice(0, MAX_DISPLAY);
                }
            }

            if (fusionInterventionHistory.length === 0) {
                list.innerHTML = '<li class="empty-message">No fusion or intervention activity yet</li>';
                return;
            }

            list.innerHTML = fusionInterventionHistory.map(activity => {
                const fusion = activity.fusion;
                const intervention = activity.intervention;

                // Determine overall status
                let overallStatus = 'UNKNOWN';
                let statusClass = 'error';

                if (fusion && intervention) {
                    // Both activities present
                    if (fusion.status === 'success' && intervention.status === 'success') {
                        overallStatus = 'SUCCESS';
                        statusClass = 'success';
                    } else if (fusion.status === 'success' || intervention.status === 'success') {
                        overallStatus = 'PARTIAL';
                        statusClass = 'partial_success';
                    } else {
                        overallStatus = 'FAILED';
                        statusClass = 'error';
                    }
                } else if (fusion) {
                    overallStatus = fusion.status === 'success' ? 'SUCCESS' : 'FAILED';
                    statusClass = fusion.status === 'success' ? 'success' : 'error';
                } else if (intervention) {
                    overallStatus = intervention.status === 'success' ? 'SUCCESS' : 'FAILED';
                    statusClass = intervention.status === 'success' ? 'success' : 'error';
                }

                return `
                    <li class="activity-item ${statusClass}">
                        <div class="item-header">
                            <span>User: ${activity.user_id.substring(0, 8)}...</span>
                            <span class="status-badge status-${statusClass === 'partial_success' ? 'partial' : statusClass}">[${overallStatus}]</span>
                        </div>

                        ${fusion ? `
                            <div style="border-left: 3px solid #ff6b6b; padding-left: 10px; margin: 8px 0;">
                                <div style="font-weight: bold; color: #ff6b6b; margin-bottom: 4px;">FUSION RESULT: ${fusion.emotion_label || 'N/A'} (${fusion.confidence_score ? (fusion.confidence_score * 100).toFixed(1) : '0.0'}%)</div>
                                <div class="item-detail timestamp">${formatTimestamp(fusion.timestamp)}</div>
                                ---
                                ${fusion.model_signals ? `
                                    <div class="item-detail" style="font-size: 0.85em; margin-top: 4px;">
                                        Signals Received: SER(${fusion.model_signals.ser || 0}) FER(${fusion.model_signals.fer || 0}) Vitals(${fusion.model_signals.vitals || 0})
                                    </div>
                                ` : ''}
                                ${fusion.model_signals_detail ? `
                                    <div class="item-detail" style="font-size: 0.8em; margin-top: 4px; color: #888;">
                                        ${fusion.model_signals_detail.ser && fusion.model_signals_detail.ser.length > 0 ? `
                                            SER: ${fusion.model_signals_detail.ser.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                        ` : ''}
                                        ${fusion.model_signals_detail.fer && fusion.model_signals_detail.fer.length > 0 ? `
                                            FER: ${fusion.model_signals_detail.fer.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                        ` : ''}
                                        ${fusion.model_signals_detail.vitals && fusion.model_signals_detail.vitals.length > 0 ? `
                                            Vitals: ${fusion.model_signals_detail.vitals.map(s => `${s.emotion_label}(${(s.confidence * 100).toFixed(0)}%)`).join(', ')}
                                        ` : ''}
                                    </div>
                                ` : ''}
                                ${fusion.fusion_calculation_log ? `
                                    <div class="item-detail" style="font-size: 0.8em; margin-top: 4px; color: #aaa; font-style: italic;">
                                        Calculation: ${fusion.fusion_calculation_log}
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}

                        ${intervention ? `
                            <div style="border-left: 3px solid #4ecdc4; padding-left: 10px; margin: 8px 0;">
                                <div style="font-weight: bold; color: #4ecdc4; margin-bottom: 4px;">INTERVENTION</div>
                                ${intervention.decision ? `
                                    <div class="item-detail" style="font-size: 0.85em;">
                                        TRIGGER: ${intervention.decision.trigger_intervention ? 'YES' : 'NO'}
                                        (Confidence: ${((intervention.decision.confidence || 0) * 100).toFixed(1)}%)
                                    </div>
                                ` : ''}
                                ${intervention.ranked_activities && intervention.ranked_activities.length > 0 ? `
                                    <div class="item-detail" style="font-size: 0.85em; margin-top: 4px;">
                                        Activities: ${intervention.ranked_activities.map(a => `${a.rank}.${a.activity_type}`).join(', ')}
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}

                        ${fusion && fusion.db_write_success !== undefined ? `
                            <div class="item-detail" style="font-size: 0.85em; margin-top: 8px;">
                                Database: ${fusion.db_write_success ? '✓ Written' : '✗ Failed'}
                            </div>
                        ` : intervention && intervention.db_write_success !== undefined ? `
                            <div class="item-detail" style="font-size: 0.85em; margin-top: 8px;">
                                Database: ${intervention.db_write_success ? '✓ Written' : '✗ Failed'}
                            </div>
                        ` : ''}

                        ${(fusion && fusion.error) || (intervention && intervention.error) ? `
                            <div class="item-detail" style="color: #ff6b6b; font-size: 0.85em; margin-top: 4px;">
                                Error: ${fusion.error || intervention.error}
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
            html += `<div>Database: <span style="color: #888;">Not monitored by dashboard</span></div>`;
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
            // Database health not monitored by this dashboard
            document.getElementById('dbHealth').className = 'health-indicator health-healthy';
        }
        
        async function fetchDashboardData() {
            try {
                const response = await fetch('/api/dashboard/status');
                const data = await response.json();

                updateModelServices(data.model_services || {});
                updateFusionIntervention({
                    fusion: data.fusion || [],
                    intervention: data.intervention || []
                });
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


async def check_model_service_health(url: str, service_name: str) -> Dict:
    """Check health of a model service via HTTP."""
    try:
        timeout = httpx.Timeout(5.0)  # 5 second timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            health_url = f"{url}/health" if not url.endswith("/health") else url
            response = await client.get(health_url)
            if response.status_code == 200:
                try:
                    data = response.json()
                    return {
                        "status": "healthy",
                        "response": data,
                        "error": None
                    }
                except:
                    return {
                        "status": "healthy",  # Service responded but not with JSON
                        "response": None,
                        "error": None
                    }
            else:
                return {
                    "status": "unhealthy",
                    "response": None,
                    "error": f"HTTP {response.status_code}"
                }
    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "response": None,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "response": None,
            "error": str(e)
        }


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
        # Database status is not monitored by this cloud service dashboard
        services["database"] = "not_monitored"
        
        # Check fusion service (assume healthy if we can read logs)
        services["fusion"] = "healthy"
        
        # Check intervention service
        services["intervention"] = "healthy"
        
        # Check context service
        services["context"] = "healthy"
        
        # Get model service status with monitoring data
        model_services_data = {}
        try:
            # Define model services with their URLs
            model_services_config = {
                "vitals": {
                    "url": "https://well-bot-bvs-emotion-520080168829.asia-south1.run.app",
                    "status": "unknown",
                    "last_activity": {},
                    "recent_signals": 0,
                    "error": None
                },
                "fer": {
                    "url": "https://wellbot-fer-backend-520080168829.asia-southeast1.run.app",
                    "status": "unknown",
                    "last_activity": {},
                    "recent_signals": 0,
                    "error": None
                },
                "ser": {
                    "url": "https://well-bot-emotionrecognition-520080168829.asia-south1.run.app",
                    "status": "unknown",
                    "last_activity": {},
                    "recent_signals": 0,
                    "error": None
                }
            }

            # Check SER service status and get detailed activity data
            try:
                # Query SER service status endpoint for real-time data
                ser_status_url = f"{model_services_config['ser']['url']}/ser/status"

                try:
                    # Get detailed status from SER service
                    timeout = httpx.Timeout(5.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(ser_status_url)

                        if response.status_code == 200:
                            ser_data = response.json()
                            model_services_config["ser"]["status"] = ser_data.get("status", "unknown")
                            model_services_config["ser"]["queue_size"] = ser_data.get("queue_size", 0)

                            # Get recent requests
                            recent_requests = ser_data.get("recent_requests", [])
                            if recent_requests:
                                model_services_config["ser"]["recent_signals"] = len(recent_requests)
                                # Get the most recent request
                                latest_request = recent_requests[0]  # Already sorted newest first
                                model_services_config["ser"]["last_activity"] = {
                                    "timestamp": latest_request.get("timestamp"),
                                    "user_id": latest_request.get("user_id"),
                                    "type": "request_received"
                                }
                            else:
                                model_services_config["ser"]["recent_signals"] = 0

                            # Get current processing info
                            current_processing = ser_data.get("current_processing")
                            if current_processing:
                                model_services_config["ser"]["current_processing"] = {
                                    "user_id": current_processing.get("user_id"),
                                    "started_at": current_processing.get("started_at"),
                                    "filename": current_processing.get("filename"),
                                    "status": "processing"
                                }

                            # Get recent results with all fields
                            recent_results = ser_data.get("recent_results", [])
                            if recent_results:
                                # Get the most recent result
                                latest_result = recent_results[0]
                                model_services_config["ser"]["last_successful_result"] = {
                                    "timestamp": latest_result.get("timestamp"),
                                    "user_id": latest_result.get("user_id"),
                                    "filename": latest_result.get("filename"),
                                    "emotion": latest_result.get("emotion"),
                                    "emotion_confidence": latest_result.get("emotion_confidence"),
                                    "sentiment": latest_result.get("sentiment"),
                                    "transcript": latest_result.get("transcript"),
                                    "language": latest_result.get("language"),
                                    "db_write_success": latest_result.get("db_write_success"),
                                    "aggregation_pending": latest_result.get("aggregation_pending", True)
                                }

                        else:
                            # Fallback to basic health check
                            model_services_config["ser"]["status"] = "unhealthy"
                            model_services_config["ser"]["error"] = f"HTTP {response.status_code}"

                except httpx.TimeoutException:
                    model_services_config["ser"]["status"] = "unhealthy"
                    model_services_config["ser"]["error"] = "Timeout"
                except Exception as api_e:
                    logger.debug(f"Could not query SER status API: {api_e}")
                    # Fallback to basic health check
                    ser_health = await check_model_service_health(
                        model_services_config["ser"]["url"], "SER"
                    )
                    model_services_config["ser"]["status"] = ser_health["status"]
                    if ser_health["error"]:
                        model_services_config["ser"]["error"] = ser_health["error"]

            except Exception as e:
                model_services_config["ser"]["error"] = str(e)
                model_services_config["ser"]["status"] = "error"

            # Check FER service status and get detailed activity data
            try:
                # Query FER service status endpoint for real-time data
                fer_status_url = f"{model_services_config['fer']['url']}/fer/status"

                try:
                    # Get detailed status from FER service
                    timeout = httpx.Timeout(5.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(fer_status_url)

                        if response.status_code == 200:
                            fer_data = response.json()
                            model_services_config["fer"]["status"] = fer_data.get("status", "unknown")

                            # Get recent requests
                            recent_requests = fer_data.get("recent_requests", [])
                            if recent_requests:
                                model_services_config["fer"]["recent_signals"] = len(recent_requests)
                                # Get the most recent request
                                latest_request = recent_requests[0]  # Already sorted newest first
                                model_services_config["fer"]["last_activity"] = {
                                    "timestamp": latest_request.get("timestamp"),
                                    "user_id": latest_request.get("user_id"),
                                    "type": "request_received"
                                }
                            else:
                                model_services_config["fer"]["recent_signals"] = 0

                            # Get recent results with all fields
                            recent_results = fer_data.get("recent_results", [])
                            if recent_results:
                                # Get the most recent result
                                latest_result = recent_results[0]
                                model_services_config["fer"]["last_successful_result"] = {
                                    "timestamp": latest_result.get("timestamp"),
                                    "user_id": latest_result.get("user_id"),
                                    "emotion": latest_result.get("emotion"),
                                    "emotion_confidence": latest_result.get("emotion_confidence"),
                                    "db_write_success": latest_result.get("db_write_success"),
                                    "aggregation_complete": latest_result.get("aggregation_complete", False)
                                }

                        else:
                            # Fallback to basic health check
                            model_services_config["fer"]["status"] = "unhealthy"
                            model_services_config["fer"]["error"] = f"HTTP {response.status_code}"

                except httpx.TimeoutException:
                    model_services_config["fer"]["status"] = "unhealthy"
                    model_services_config["fer"]["error"] = "Timeout"
                except Exception as api_e:
                    logger.debug(f"Could not query FER status API: {api_e}")
                    # Fallback to basic health check
                    fer_health = await check_model_service_health(
                        model_services_config["fer"]["url"], "FER"
                    )
                    model_services_config["fer"]["status"] = fer_health["status"]
                    if fer_health["error"]:
                        model_services_config["fer"]["error"] = fer_health["error"]

            except Exception as e:
                model_services_config["fer"]["error"] = str(e)
                model_services_config["fer"]["status"] = "error"

            # Check Vitals service health
            try:
                vitals_health = await check_model_service_health(
                    model_services_config["vitals"]["url"], "Vitals"
                )
                model_services_config["vitals"]["status"] = vitals_health["status"]
                if vitals_health["error"]:
                    model_services_config["vitals"]["error"] = vitals_health["error"]

            except Exception as e:
                model_services_config["vitals"]["error"] = str(e)
                model_services_config["vitals"]["status"] = "error"

            model_services_data = model_services_config

        except Exception as e:
            logger.warning(f"Error getting model services data: {e}")
        
        # Calculate recent stats (last hour)
        now = datetime.now(get_malaysia_timezone())
        one_hour_ago = datetime.fromtimestamp(now.timestamp() - 3600, tz=now.tzinfo)
        
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
            "model_services": model_services_data,
            "status": {
                "services": services,
                "model_services": {},  # Legacy format - keeping empty for backward compatibility
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
            "model_services": {},
            "status": {
                "services": {},
                "model_services": {},
                "stats": {}
            },
            "error": str(e)
        }

