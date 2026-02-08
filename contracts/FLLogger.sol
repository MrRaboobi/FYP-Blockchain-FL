// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FLLogger
 * @dev Logs federated learning model updates for provenance
 */
contract FLLogger {
    
    // Structure to store update information
    struct ModelUpdate {
        uint256 round;
        uint256 clientId;
        bytes32 modelHash;
        uint256 dataSize;
        uint256 timestamp;
        uint256 accuracy;  // Stored as integer (e.g., 9580 = 95.80%)
    }
    
    // Array to store all updates
    ModelUpdate[] public updates;
    
    // Mapping: round => array of update indices
    mapping(uint256 => uint256[]) public roundUpdates;
    
    // Mapping: clientId => array of update indices
    mapping(uint256 => uint256[]) public clientUpdates;
    
    // Events
    event UpdateLogged(
        uint256 indexed round,
        uint256 indexed clientId,
        bytes32 modelHash,
        uint256 timestamp
    );
    
    event RoundCompleted(
        uint256 indexed round,
        uint256 numClients,
        uint256 timestamp
    );
    
    /**
     * @dev Log a model update from a client
     * @param _round Training round number
     * @param _clientId Client identifier
     * @param _modelHash Hash of model weights
     * @param _dataSize Number of training samples
     * @param _accuracy Accuracy as integer (e.g., 9580 = 95.80%)
     */
    function logUpdate(
        uint256 _round,
        uint256 _clientId,
        bytes32 _modelHash,
        uint256 _dataSize,
        uint256 _accuracy
    ) public {
        
        // Create update record
        ModelUpdate memory newUpdate = ModelUpdate({
            round: _round,
            clientId: _clientId,
            modelHash: _modelHash,
            dataSize: _dataSize,
            timestamp: block.timestamp,
            accuracy: _accuracy
        });
        
        // Store update
        uint256 updateIndex = updates.length;
        updates.push(newUpdate);
        
        // Index by round and client
        roundUpdates[_round].push(updateIndex);
        clientUpdates[_clientId].push(updateIndex);
        
        // Emit event
        emit UpdateLogged(_round, _clientId, _modelHash, block.timestamp);
    }
    
    /**
     * @dev Mark a round as completed
     * @param _round Training round number
     */
    function completeRound(uint256 _round) public {
        uint256 numClients = roundUpdates[_round].length;
        emit RoundCompleted(_round, numClients, block.timestamp);
    }
    
    /**
     * @dev Get total number of updates
     */
    function getTotalUpdates() public view returns (uint256) {
        return updates.length;
    }
    
    /**
     * @dev Get all updates for a specific round
     * @param _round Training round number
     */
    function getRoundUpdates(uint256 _round) public view returns (uint256[] memory) {
        return roundUpdates[_round];
    }
    
    /**
     * @dev Get all updates from a specific client
     * @param _clientId Client identifier
     */
    function getClientUpdates(uint256 _clientId) public view returns (uint256[] memory) {
        return clientUpdates[_clientId];
    }
    
    /**
     * @dev Get update details by index
     * @param _index Update index
     */
    function getUpdate(uint256 _index) public view returns (
        uint256 round,
        uint256 clientId,
        bytes32 modelHash,
        uint256 dataSize,
        uint256 timestamp,
        uint256 accuracy
    ) {
        require(_index < updates.length, "Update does not exist");
        ModelUpdate memory update = updates[_index];
        return (
            update.round,
            update.clientId,
            update.modelHash,
            update.dataSize,
            update.timestamp,
            update.accuracy
        );
    }
    // ==========================================
    // SYNTHETIC DATA GOVERNANCE (Phase 4)
    // ==========================================
    
    struct SyntheticRequest {
        uint256 requestId;
        uint256 clientId;
        uint256 classLabel;
        uint256 quantity;
        bool approved;
        bool generated;
        uint256 timestamp;
    }
    
    SyntheticRequest[] public syntheticRequests;
    
    mapping(uint256 => uint256[]) public clientSyntheticRequests;  // clientId => requestIds
    mapping(uint256 => uint256) public syntheticQuota;  // clientId => remaining quota
    
    event SyntheticRequested(
        uint256 indexed requestId,
        uint256 indexed clientId,
        uint256 classLabel,
        uint256 quantity,
        uint256 timestamp
    );
    
    event SyntheticApproved(
        uint256 indexed requestId,
        uint256 indexed clientId,
        uint256 quantity,
        uint256 timestamp
    );
    
    event SyntheticGenerated(
        uint256 indexed requestId,
        uint256 indexed clientId,
        uint256 classLabel,
        uint256 quantity,
        uint256 timestamp
    );
    
    /**
     * @dev Initialize synthetic data quota for a client
     * @param _clientId Client identifier
     * @param _quota Maximum synthetic samples allowed
     */
    function setSyntheticQuota(uint256 _clientId, uint256 _quota) public {
        syntheticQuota[_clientId] = _quota;
    }
    
    /**
     * @dev Request synthetic data generation
     * @param _clientId Client identifier
     * @param _classLabel Class to generate samples for
     * @param _quantity Number of samples requested
     */
    function requestSynthetic(
        uint256 _clientId,
        uint256 _classLabel,
        uint256 _quantity
    ) public returns (uint256) {
        
        uint256 requestId = syntheticRequests.length;
        
        SyntheticRequest memory newRequest = SyntheticRequest({
            requestId: requestId,
            clientId: _clientId,
            classLabel: _classLabel,
            quantity: _quantity,
            approved: false,
            generated: false,
            timestamp: block.timestamp
        });
        
        syntheticRequests.push(newRequest);
        clientSyntheticRequests[_clientId].push(requestId);
        
        emit SyntheticRequested(requestId, _clientId, _classLabel, _quantity, block.timestamp);
        
        return requestId;
    }
    
    /**
     * @dev Approve synthetic data request
     * @param _requestId Request identifier
     */
    function approveSynthetic(uint256 _requestId) public {
        require(_requestId < syntheticRequests.length, "Request does not exist");
        
        SyntheticRequest storage request = syntheticRequests[_requestId];
        require(!request.approved, "Already approved");
        
        // Check quota
        uint256 clientId = request.clientId;
        require(syntheticQuota[clientId] >= request.quantity, "Quota exceeded");
        
        // Approve
        request.approved = true;
        syntheticQuota[clientId] -= request.quantity;
        
        emit SyntheticApproved(_requestId, clientId, request.quantity, block.timestamp);
    }
    
    /**
     * @dev Mark synthetic data as generated
     * @param _requestId Request identifier
     */
    function markSyntheticGenerated(uint256 _requestId) public {
        require(_requestId < syntheticRequests.length, "Request does not exist");
        
        SyntheticRequest storage request = syntheticRequests[_requestId];
        require(request.approved, "Not approved");
        require(!request.generated, "Already generated");
        
        request.generated = true;
        
        emit SyntheticGenerated(
            _requestId,
            request.clientId,
            request.classLabel,
            request.quantity,
            block.timestamp
        );
    }
    
    /**
     * @dev Get synthetic request details
     * @param _requestId Request identifier
     */
    function getSyntheticRequest(uint256 _requestId) public view returns (
        uint256 clientId,
        uint256 classLabel,
        uint256 quantity,
        bool approved,
        bool generated,
        uint256 timestamp
    ) {
        require(_requestId < syntheticRequests.length, "Request does not exist");
        SyntheticRequest memory request = syntheticRequests[_requestId];
        return (
            request.clientId,
            request.classLabel,
            request.quantity,
            request.approved,
            request.generated,
            request.timestamp
        );
    }
    
    /**
     * @dev Get total synthetic requests
     */
    function getTotalSyntheticRequests() public view returns (uint256) {
        return syntheticRequests.length;
    }
    
    /**
     * @dev Get client's remaining quota
     */
    function getQuota(uint256 _clientId) public view returns (uint256) {
        return syntheticQuota[_clientId];
    }
}

