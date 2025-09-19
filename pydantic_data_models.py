from pydantic import BaseModel, Field
from typing import TypedDict, List

class GraphState(TypedDict):
    image_bytes: bytes
    ocr_text: str
    document_type: str
    extracted_json: dict

class MedicalReport(BaseModel):
    patient_name: str = Field(description="The name of the patient.")
    hospital_name: str = Field(description="The name of the hospital or clinic.")
    report_date: str = Field(description="The date the report was issued.")
    report_type: str = Field(description="The type of the report (e.g., 'Blood Test', 'MRI Scan').")
    clinical_findings: str = Field(description="A summary of the clinical findings or results.")
    extraction_confidence: float = Field(description="Confidence score from 0.0 to 1.0 on the accuracy of the extracted data.")

class Prescription(BaseModel):
    patient_name: str = Field(description="The name of the patient.")
    doctor_name: str = Field(description="The name of the prescribing doctor.")
    clinic_name: str = Field(description="The name of the clinic or hospital.")
    prescription_date: str = Field(description="The date the prescription was issued.")
    diagnosis_notes: str = Field(description="Brief notes on the diagnosis or reason for the prescription.")
    extraction_confidence: float = Field(description="Confidence score from 0.0 to 1.0 on the accuracy of the extracted data.")

class Bill(BaseModel):
    patient_name: str = Field(description="The name of the patient.")
    hospital_or_clinic_name: str = Field(description="The name of the hospital or clinic.")
    bill_date: str = Field(description="The date the bill was issued.")
    total_amount: str = Field(description="The total amount due on the bill.")
    extraction_confidence: float = Field(description="Confidence score from 0.0 to 1.0 on the accuracy of the extracted data.")
    
