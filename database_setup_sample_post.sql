-- Legal Case Management Database Setup
-- Case #2024-PI-001: John Smith vs. Sarah Johnson/ABC Insurance

-- Note: The CREATE DATABASE and USE commands are for setting up from a fresh start.
-- If the database already exists, comment out the CREATE DATABASE line and connect to it directly with `psql -d legal_case_management`

-- CREATE DATABASE legal_case_management;
-- Do not include 'USE' as it is a MySQL command. Connect using psql -d legal_case_management

-- Cases table
CREATE TABLE cases (
    case_id VARCHAR(50) PRIMARY KEY,
    case_type VARCHAR(100) NOT NULL,
    date_filed DATE NOT NULL,
    status VARCHAR(50) NOT NULL,
    attorney_id INT,
    case_summary TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger function for automatic update of updated_date
CREATE OR REPLACE FUNCTION update_updated_date()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_date = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to call the function before an update on cases table
CREATE TRIGGER update_cases_updated_date
BEFORE UPDATE ON cases
FOR EACH ROW
EXECUTE PROCEDURE update_updated_date();


-- Parties table  
CREATE TABLE parties (
    party_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    party_type VARCHAR(50) NOT NULL, -- 'plaintiff', 'defendant', 'witness', 'insurance_company'
    name VARCHAR(200) NOT NULL,
    contact_info JSONB,
    insurance_info JSONB,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Documents table
CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    doc_category VARCHAR(100) NOT NULL, -- 'medical', 'financial', 'correspondence', 'police_report'
    upload_date DATE NOT NULL,
    document_title VARCHAR(300),
    metadata JSONB,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Case_events table
CREATE TABLE case_events (
    event_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    event_type VARCHAR(100) NOT NULL, -- 'accident', 'medical_treatment', 'expense', 'correspondence'
    description TEXT NOT NULL,
    amount DECIMAL(10,2), -- for expenses, damages, bills
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Insert sample data for Case #2024-PI-001

-- Insert case record
INSERT INTO cases (case_id, case_type, date_filed, status, attorney_id, case_summary) VALUES 
(
    '2024-PI-001',
    'Personal Injury - Motor Vehicle Accident',
    '2024-03-15',
    'Active - Demand Phase',
    101,
    'Rear-end collision case where plaintiff John Smith was struck by defendant Sarah Johnson while stopped at traffic light. Clear liability with admitted fault by defendant. Plaintiff sustained soft tissue injuries to back, neck, and shoulder requiring 12 weeks of medical treatment. Total medical expenses $8,950, lost wages $4,940. Defendant was distracted by cell phone use at time of accident.'
);

-- Insert parties
INSERT INTO parties (case_id, party_type, name, contact_info, insurance_info) VALUES 
(
    '2024-PI-001',
    'plaintiff',
    'John Smith',
    '{"phone": "(555) 987-6543", "address": "456 Oak Street, Springfield, CA 90210", "email": "jsmith@email.com", "dob": "1985-06-15"}',
    '{"company": "State Farm", "policy": "SF-789456123", "adjuster": "Mary Johnson", "phone": "(800) 555-1234"}'
),
(
    '2024-PI-001',
    'defendant',
    'Sarah Johnson',
    '{"phone": "(555) 456-7890", "address": "789 Pine Avenue, Springfield, CA 90215", "dob": "1992-02-22"}',
    '{"company": "ABC Insurance Company", "policy": "ABC-456789012", "adjuster": "Michael Rodriguez", "phone": "(800) 555-0199"}'
),
(
    '2024-PI-001',
    'insurance_company',
    'ABC Insurance Company',
    '{"phone": "(800) 555-0199", "address": "1500 Insurance Plaza, Los Angeles, CA 90015", "claims_dept": "claims@abcinsurance.com"}',
    '{"limits": "100/300/50", "status": "active", "coverage_confirmed": true}'
),
(
    '2024-PI-001',
    'witness',
    'Maria Rodriguez',
    '{"phone": "(555) 321-6547", "address": "234 Elm Street, Springfield, CA 90210"}',
    '{}'
),
(
    '2024-PI-001',
    'witness',
    'Robert Chen',
    '{"phone": "(555) 654-3210", "address": "567 Maple Drive, Springfield, CA 90212"}',
    '{}'
);

-- Insert documents
INSERT INTO documents (case_id, file_path, doc_category, upload_date, document_title, metadata) VALUES 
(
    '2024-PI-001',
    '/sample_docs/2024-PI-001/medical_records_dr_jones.pdf',
    'medical',
    '2024-03-20',
    'Medical Records - Dr. Michael Jones - Central Valley Medical Group',
    '{"pages": 8, "date_range": "2024-03-16 to 2024-06-10", "provider": "Dr. Michael Jones", "total_expenses": 8950, "injuries": ["lumbar strain", "cervical whiplash", "shoulder contusion"]}'
),
(
    '2024-PI-001',
    '/sample_docs/2024-PI-001/police_report_incident_789.pdf',
    'police_report',
    '2024-03-16',
    'Police Report - SPD Report #789 - Traffic Collision',
    '{"report_number": "SPD-2024-789", "officer": "Jennifer Martinez", "fault_determination": "100% defendant", "citations": ["CVC 21703", "CVC 23123"], "total_fines": 400}'
),
(
    '2024-PI-001',
    '/sample_docs/2024-PI-001/wage_statements_2024.pdf',
    'financial',
    '2024-07-05',
    'Wage Statements and Employment Records - Pacific Construction',
    '{"employer": "Pacific Construction Company", "position": "Construction Supervisor", "hourly_rate": 32.50, "total_wage_loss": 4940, "time_off_weeks": 10.5}'
),
(
    '2024-PI-001',
    '/sample_docs/2024-PI-001/insurance_correspondence.pdf',
    'correspondence',
    '2024-06-25',
    'Insurance Correspondence - ABC Insurance Communications',
    '{"claim_number": "ABC-2024-789456", "adjuster": "Michael Rodriguez", "settlement_range": "25000-35000", "liability_accepted": true}'
);

-- Insert case events
INSERT INTO case_events (case_id, event_date, event_type, description, amount) VALUES 
(
    '2024-PI-001',
    '2024-03-15',
    'accident',
    'Motor vehicle accident occurred at 15:30 hours at intersection of Main Street and First Avenue. Plaintiff rear-ended by defendant while stopped at red light. Defendant admits to cell phone distraction.',
    NULL
),
(
    '2024-PI-001',
    '2024-03-15',
    'medical_treatment',
    'Plaintiff transported by ambulance to Springfield General Hospital. Initial ER evaluation for back, neck, and shoulder injuries.',
    1200.00
),
(
    '2024-PI-001',
    '2024-03-16',
    'medical_treatment',
    'Initial examination with Dr. Michael Jones. Diagnosed with lumbar strain, cervical whiplash, and right shoulder contusion. Work restrictions imposed.',
    225.00
),
(
    '2024-PI-001',
    '2024-03-25',
    'medical_treatment',
    'MRI of lumbar spine showing mild disc bulge L4-L5. Physical therapy ordered 3x per week.',
    1400.00
),
(
    '2024-PI-001',
    '2024-03-30',
    'medical_treatment',
    'Follow-up with Dr. Jones. Persistent pain, sleep disruption. Gabapentin added for nerve pain. Work restrictions extended.',
    225.00
),
(
    '2024-PI-001',
    '2024-04-20',
    'medical_treatment',
    'Gradual improvement noted. Physical therapy reduced to 2x per week. Begin light duty work return.',
    225.00
),
(
    '2024-PI-001',
    '2024-06-10',
    'medical_treatment',
    'Final evaluation with Dr. Jones. Maximum medical improvement reached. 5% permanent partial impairment. Return to work with lifting restrictions.',
    225.00
),
(
    '2024-PI-001',
    '2024-03-16',
    'expense',
    'Ambulance transport from accident scene to Springfield General Hospital.',
    350.00
),
(
    '2024-PI-001',
    '2024-03-16',
    'expense',
    'Vehicle towing and storage fees - Jerry''s Towing Service.',
    285.00
),
(
    '2024-PI-001',
    '2024-03-20',
    'expense',
    'Prescription medications - pain medication, muscle relaxants, anti-inflammatory.',
    127.50
),
(
    '2024-PI-001',
    '2024-04-15',
    'expense',
    'Physical therapy sessions (12 sessions) - Central Valley Physical Therapy.',
    1440.00
),
(
    '2024-PI-001',
    '2024-05-20',
    'expense',
    'Additional physical therapy sessions (12 sessions) - Central Valley Physical Therapy.',
    1440.00
),
(
    '2024-PI-001',
    '2024-06-15',
    'expense',
    'Additional prescription medications and medical supplies.',
    442.50
),
(
    '2024-PI-001',
    '2024-03-16',
    'correspondence',
    'Initial claim report filed with ABC Insurance Company. Claim number ABC-2024-789456 assigned.',
    NULL
),
(
    '2024-PI-001',
    '2024-03-25',
    'correspondence',
    'Property damage settlement offer received from ABC Insurance for $12,947.83.',
    12947.83
),
(
    '2024-PI-001',
    '2024-05-20',
    'correspondence',
    'Settlement authority letter received from ABC Insurance indicating willingness to settle bodily injury claim in range of $25,000-$35,000.',
    NULL
),
(
    '2024-PI-001',
    '2024-06-25',
    'correspondence',
    'Counter-offer received from ABC Insurance for $28,500 in response to demand letter.',
    28500.00
);

-- Create indexes for better query performance
CREATE INDEX idx_cases_status ON cases(status);
CREATE INDEX idx_cases_type ON cases(case_type);
CREATE INDEX idx_parties_case_id ON parties(case_id);
CREATE INDEX idx_parties_type ON parties(party_type);
CREATE INDEX idx_documents_case_id ON documents(case_id);
CREATE INDEX idx_documents_category ON documents(doc_category);
CREATE INDEX idx_events_case_id ON case_events(case_id);
CREATE INDEX idx_events_type ON case_events(event_type);
CREATE INDEX idx_events_date ON case_events(event_date);

-- Sample queries for testing MCP functions

-- Get case details with all parties
SELECT 
    c.*,
    p.party_type,
    p.name as party_name,
    p.contact_info,
    p.insurance_info
FROM cases c
LEFT JOIN parties p ON c.case_id = p.case_id
WHERE c.case_id = '2024-PI-001';

-- Get documents by category
SELECT * FROM documents 
WHERE case_id = '2024-PI-001' 
AND doc_category = 'medical';

-- Get case timeline ordered by date
SELECT 
    event_date,
    event_type,
    description,
    amount
FROM case_events 
WHERE case_id = '2024-PI-001'
ORDER BY event_date;

-- Calculate financial summary
SELECT 
    event_type,
    SUM(amount) as total_amount,
    COUNT(*) as event_count
FROM case_events 
WHERE case_id = '2024-PI-001' 
AND amount IS NOT NULL
GROUP BY event_type
ORDER BY total_amount DESC;

-- Get total medical expenses
SELECT SUM(amount) as total_medical_expenses
FROM case_events 
WHERE case_id = '2024-PI-001' 
AND event_type IN ('medical_treatment', 'expense')
AND description LIKE '%medical%' OR description LIKE '%therapy%' OR description LIKE '%prescription%';

-- Get party details by type
SELECT * FROM parties 
WHERE case_id = '2024-PI-001' 
AND party_type = 'plaintiff';
