%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4CAF50',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#2E7D32',
  'lineColor': '#555',
  'secondaryColor': '#E3F2FD',
  'tertiaryColor': '#FFF'
}}}%%

flowchart TD
    A([START]):::start --> B[Sensor samples vitals every 5s]
    B --> C[Validate reading]
    
    C --> D{Error?}
    
    D -- YES --> E[Log sensor fault]
    E --> F[Skip transmission]
    F --> B
    
    D -- NO --> G[Package JSON]
    G --> H[Publish to MQTT topic]
    
    H --> I{Threshold breach?}
    
    I -- YES --> J[Trigger alert<br/>Dashboard + SMS]:::alert
    J --> B
    
    I -- NO --> K[Write to database]
    K --> L[Update dashboard display]
    L --> B

    %% Styling classes
    classDef start fill:#2E7D32,color:#fff,font-weight:bold;
    classDef alert fill:#D32F2F,color:#fff,font-weight:bold;
