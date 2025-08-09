"""
Global Compliance Module for DP-Federated LoRA system.

This module implements comprehensive compliance capabilities for international
regulations including GDPR, CCPA, PDPA, and other regional privacy laws,
providing automated compliance checking, data governance, and audit trails
for federated learning deployments worldwide.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import uuid

from .config import FederatedConfig, PrivacyConfig
from .i18n import SupportedLanguage, i18n_manager


logger = logging.getLogger(__name__)


class PrivacyRegulation(Enum):
    """Supported privacy regulations worldwide."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA_SINGAPORE = "pdpa_sg"  # Personal Data Protection Act (Singapore)
    PDPA_THAILAND = "pdpa_th"  # Personal Data Protection Act (Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act_aus"  # Privacy Act (Australia)
    APPI = "appi"  # Act on Protection of Personal Information (Japan)
    POPIA = "popia"  # Protection of Personal Information Act (South Africa)


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REQUIRES_ACTION = "requires_action"


class DataProcessingPurpose(Enum):
    """Lawful purposes for data processing under various regulations."""
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONSENT = "consent"
    CONTRACT_PERFORMANCE = "contract_performance"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    SCIENTIFIC_RESEARCH = "scientific_research"


@dataclass
class RegulationRequirements:
    """Requirements for a specific privacy regulation."""
    
    name: str
    jurisdiction: str
    consent_required: bool
    purpose_limitation: bool
    data_minimization: bool
    storage_limitation: bool
    accuracy_requirement: bool
    security_requirement: bool
    breach_notification_hours: int
    right_to_erasure: bool
    right_to_portability: bool
    right_to_rectification: bool
    privacy_by_design: bool
    dpo_required: bool  # Data Protection Officer
    impact_assessment_required: bool
    cross_border_transfer_restrictions: bool
    min_age_consent: int
    max_fine_percentage: float
    max_fine_amount: int


@dataclass
class ComplianceRecord:
    """Record of compliance assessment."""
    
    record_id: str
    regulation: PrivacyRegulation
    assessment_date: datetime
    status: ComplianceStatus
    score: float
    requirements_met: List[str]
    requirements_failed: List[str]
    recommendations: List[str]
    auditor: Optional[str] = None
    next_review_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSubjectRequest:
    """Data subject rights request."""
    
    request_id: str
    subject_id: str
    request_type: str  # access, rectification, erasure, portability, restriction
    regulation: PrivacyRegulation
    submission_date: datetime
    status: str  # pending, processing, completed, rejected
    response_deadline: datetime
    justification: Optional[str] = None
    processed_by: Optional[str] = None
    completion_date: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None


class RegulationDatabase:
    """Database of privacy regulation requirements."""
    
    @staticmethod
    def get_regulation_requirements() -> Dict[PrivacyRegulation, RegulationRequirements]:
        """Get comprehensive regulation requirements."""
        return {
            PrivacyRegulation.GDPR: RegulationRequirements(
                name="General Data Protection Regulation",
                jurisdiction="European Union",
                consent_required=True,
                purpose_limitation=True,
                data_minimization=True,
                storage_limitation=True,
                accuracy_requirement=True,
                security_requirement=True,
                breach_notification_hours=72,
                right_to_erasure=True,
                right_to_portability=True,
                right_to_rectification=True,
                privacy_by_design=True,
                dpo_required=True,
                impact_assessment_required=True,
                cross_border_transfer_restrictions=True,
                min_age_consent=16,
                max_fine_percentage=4.0,
                max_fine_amount=20000000
            ),
            
            PrivacyRegulation.CCPA: RegulationRequirements(
                name="California Consumer Privacy Act",
                jurisdiction="California, USA",
                consent_required=False,  # Opt-out model
                purpose_limitation=True,
                data_minimization=True,
                storage_limitation=False,
                accuracy_requirement=False,
                security_requirement=True,
                breach_notification_hours=0,  # No specific requirement
                right_to_erasure=True,
                right_to_portability=True,
                right_to_rectification=False,
                privacy_by_design=False,
                dpo_required=False,
                impact_assessment_required=False,
                cross_border_transfer_restrictions=False,
                min_age_consent=13,
                max_fine_percentage=0.0,
                max_fine_amount=7500  # Per violation
            ),
            
            PrivacyRegulation.PDPA_SINGAPORE: RegulationRequirements(
                name="Personal Data Protection Act (Singapore)",
                jurisdiction="Singapore",
                consent_required=True,
                purpose_limitation=True,
                data_minimization=True,
                storage_limitation=True,
                accuracy_requirement=True,
                security_requirement=True,
                breach_notification_hours=72,
                right_to_erasure=False,
                right_to_portability=True,
                right_to_rectification=True,
                privacy_by_design=True,
                dpo_required=True,
                impact_assessment_required=False,
                cross_border_transfer_restrictions=True,
                min_age_consent=13,
                max_fine_percentage=0.0,
                max_fine_amount=1000000  # SGD
            ),
            
            PrivacyRegulation.LGPD: RegulationRequirements(
                name="Lei Geral de Proteção de Dados",
                jurisdiction="Brazil",
                consent_required=True,
                purpose_limitation=True,
                data_minimization=True,
                storage_limitation=True,
                accuracy_requirement=True,
                security_requirement=True,
                breach_notification_hours=72,
                right_to_erasure=True,
                right_to_portability=True,
                right_to_rectification=True,
                privacy_by_design=True,
                dpo_required=True,
                impact_assessment_required=True,
                cross_border_transfer_restrictions=True,
                min_age_consent=16,
                max_fine_percentage=2.0,
                max_fine_amount=50000000  # BRL
            )
        }


class ComplianceEngine:
    """Main compliance engine for privacy regulations."""
    
    def __init__(self, config: FederatedConfig):
        """Initialize compliance engine."""
        self.config = config
        self.regulation_db = RegulationDatabase()
        self.requirements = self.regulation_db.get_regulation_requirements()
        
        # Compliance state
        self.compliance_records: List[ComplianceRecord] = []
        self.data_subject_requests: List[DataSubjectRequest] = []
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Active regulations based on deployment regions
        self.active_regulations: Set[PrivacyRegulation] = set()
        
        logger.info("Compliance engine initialized")
    
    def enable_regulation(self, regulation: PrivacyRegulation):
        """Enable compliance for a specific regulation."""
        self.active_regulations.add(regulation)
        logger.info(f"Enabled compliance for {regulation.value}")
        
        # Log compliance activation
        self._log_audit_event("regulation_enabled", {
            "regulation": regulation.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def assess_compliance(
        self,
        regulation: PrivacyRegulation,
        system_config: Dict[str, Any]
    ) -> ComplianceRecord:
        """Assess compliance with a specific regulation."""
        logger.info(f"Assessing compliance with {regulation.value}")
        
        requirements = self.requirements[regulation]
        requirements_met = []
        requirements_failed = []
        recommendations = []
        
        # Check each requirement
        if requirements.consent_required:
            if system_config.get("consent_mechanism_enabled", False):
                requirements_met.append("consent_mechanism")
            else:
                requirements_failed.append("consent_mechanism")
                recommendations.append("Implement explicit consent mechanism for data processing")
        
        if requirements.data_minimization:
            if system_config.get("data_minimization_enabled", False):
                requirements_met.append("data_minimization")
            else:
                requirements_failed.append("data_minimization")
                recommendations.append("Implement data minimization techniques")
        
        if requirements.security_requirement:
            if system_config.get("encryption_enabled", False):
                requirements_met.append("encryption")
            else:
                requirements_failed.append("encryption")
                recommendations.append("Enable end-to-end encryption for data protection")
        
        if requirements.privacy_by_design:
            # Check if differential privacy is properly configured
            privacy_config = system_config.get("privacy_config", {})
            epsilon = privacy_config.get("epsilon", 0)
            
            if epsilon > 0 and epsilon <= 10:  # Reasonable privacy budget
                requirements_met.append("privacy_by_design")
            else:
                requirements_failed.append("privacy_by_design")
                recommendations.append("Configure appropriate differential privacy parameters")
        
        if requirements.breach_notification_hours > 0:
            if system_config.get("breach_notification_system", False):
                requirements_met.append("breach_notification")
            else:
                requirements_failed.append("breach_notification")
                recommendations.append(f"Implement breach notification system (within {requirements.breach_notification_hours} hours)")
        
        if requirements.dpo_required:
            if system_config.get("dpo_appointed", False):
                requirements_met.append("data_protection_officer")
            else:
                requirements_failed.append("data_protection_officer")
                recommendations.append("Appoint a Data Protection Officer")
        
        # Calculate compliance score
        total_requirements = len(requirements_met) + len(requirements_failed)
        compliance_score = len(requirements_met) / total_requirements if total_requirements > 0 else 0.0
        
        # Determine compliance status
        if compliance_score >= 0.95:
            status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 0.80:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        elif compliance_score >= 0.60:
            status = ComplianceStatus.REQUIRES_ACTION
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Create compliance record
        record = ComplianceRecord(
            record_id=str(uuid.uuid4()),
            regulation=regulation,
            assessment_date=datetime.now(timezone.utc),
            status=status,
            score=compliance_score,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            auditor="automated_assessment",
            next_review_date=datetime.now(timezone.utc).replace(month=datetime.now().month + 3),  # 3 months
            metadata={
                "system_config": system_config,
                "regulation_name": requirements.name,
                "jurisdiction": requirements.jurisdiction
            }
        )
        
        self.compliance_records.append(record)
        
        # Log compliance assessment
        self._log_audit_event("compliance_assessment", {
            "regulation": regulation.value,
            "status": status.value,
            "score": compliance_score,
            "requirements_failed": len(requirements_failed)
        })
        
        return record
    
    def process_data_subject_request(
        self,
        subject_id: str,
        request_type: str,
        regulation: PrivacyRegulation,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> DataSubjectRequest:
        """Process a data subject rights request."""
        logger.info(f"Processing {request_type} request for subject {subject_id} under {regulation.value}")
        
        # Calculate response deadline based on regulation
        requirements = self.requirements[regulation]
        if regulation == PrivacyRegulation.GDPR:
            deadline_days = 30  # 1 month under GDPR
        elif regulation == PrivacyRegulation.CCPA:
            deadline_days = 45  # 45 days under CCPA
        else:
            deadline_days = 30  # Default
        
        response_deadline = datetime.now(timezone.utc).replace(
            day=datetime.now().day + deadline_days
        )
        
        # Create request record
        request = DataSubjectRequest(
            request_id=str(uuid.uuid4()),
            subject_id=hashlib.sha256(subject_id.encode()).hexdigest(),  # Hash for privacy
            request_type=request_type,
            regulation=regulation,
            submission_date=datetime.now(timezone.utc),
            status="pending",
            response_deadline=response_deadline,
            justification=additional_data.get("justification") if additional_data else None
        )
        
        self.data_subject_requests.append(request)
        
        # Log request processing
        self._log_audit_event("data_subject_request", {
            "request_id": request.request_id,
            "request_type": request_type,
            "regulation": regulation.value,
            "deadline": response_deadline.isoformat()
        })
        
        return request
    
    def generate_privacy_notice(
        self,
        regulation: PrivacyRegulation,
        language: SupportedLanguage = SupportedLanguage.ENGLISH
    ) -> str:
        """Generate a privacy notice compliant with specific regulation."""
        requirements = self.requirements[regulation]
        
        # Set language for i18n
        i18n_manager.set_language(language)
        
        notice_sections = []
        
        # Header
        notice_sections.append(f"# Privacy Notice - {requirements.name}")
        notice_sections.append(f"Jurisdiction: {requirements.jurisdiction}")
        notice_sections.append("")
        
        # Data processing purpose
        notice_sections.append("## Data Processing")
        notice_sections.append(i18n_manager.get_data_processing_notice())
        notice_sections.append("")
        
        # Consent information
        if requirements.consent_required:
            notice_sections.append("## Consent")
            notice_sections.append(i18n_manager.get_privacy_consent())
            notice_sections.append("")
        
        # Data subject rights
        notice_sections.append("## Your Rights")
        rights_text = []
        
        if requirements.right_to_erasure:
            rights_text.append("- Right to erasure (deletion) of your personal data")
        
        if requirements.right_to_portability:
            rights_text.append("- Right to data portability")
        
        if requirements.right_to_rectification:
            rights_text.append("- Right to rectification of inaccurate data")
        
        notice_sections.extend(rights_text)
        notice_sections.append("")
        
        # Security measures
        notice_sections.append("## Security")
        notice_sections.append("This system implements differential privacy and advanced encryption to protect your data.")
        notice_sections.append("")
        
        # Contact information
        notice_sections.append("## Contact")
        if requirements.dpo_required:
            notice_sections.append("Data Protection Officer: [Contact Information]")
        notice_sections.append("Privacy Questions: [Contact Information]")
        
        return "\n".join(notice_sections)
    
    def check_cross_border_transfer_compliance(
        self,
        source_region: str,
        destination_region: str,
        regulation: PrivacyRegulation
    ) -> Tuple[bool, List[str]]:
        """Check if cross-border data transfer is compliant."""
        requirements = self.requirements[regulation]
        
        if not requirements.cross_border_transfer_restrictions:
            return True, []
        
        issues = []
        
        # GDPR adequacy decisions
        if regulation == PrivacyRegulation.GDPR:
            adequate_countries = {
                "andorra", "argentina", "canada", "faroe_islands", "guernsey",
                "israel", "isle_of_man", "japan", "jersey", "new_zealand",
                "south_korea", "switzerland", "united_kingdom", "uruguay"
            }
            
            if destination_region.lower() not in adequate_countries:
                issues.append("Transfer requires additional safeguards (e.g., SCCs, BCRs)")
        
        # CCPA considerations
        elif regulation == PrivacyRegulation.CCPA:
            if destination_region.lower() not in ["usa", "united_states"]:
                issues.append("Cross-border transfer may require disclosure in privacy policy")
        
        is_compliant = len(issues) == 0
        return is_compliant, issues
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "report_id": str(uuid.uuid4()),
            "generation_date": datetime.now(timezone.utc).isoformat(),
            "active_regulations": [reg.value for reg in self.active_regulations],
            "compliance_summary": {},
            "overall_status": ComplianceStatus.COMPLIANT.value,
            "recommendations": [],
            "data_subject_requests": {
                "total": len(self.data_subject_requests),
                "pending": len([r for r in self.data_subject_requests if r.status == "pending"]),
                "overdue": len([
                    r for r in self.data_subject_requests 
                    if r.response_deadline < datetime.now(timezone.utc) and r.status != "completed"
                ])
            },
            "audit_events": len(self.audit_trail)
        }
        
        # Compliance summary by regulation
        for regulation in self.active_regulations:
            recent_records = [
                r for r in self.compliance_records 
                if r.regulation == regulation
            ]
            
            if recent_records:
                latest_record = max(recent_records, key=lambda x: x.assessment_date)
                report["compliance_summary"][regulation.value] = {
                    "status": latest_record.status.value,
                    "score": latest_record.score,
                    "last_assessment": latest_record.assessment_date.isoformat(),
                    "requirements_failed": len(latest_record.requirements_failed),
                    "next_review": latest_record.next_review_date.isoformat() if latest_record.next_review_date else None
                }
                
                # Collect recommendations
                report["recommendations"].extend(latest_record.recommendations)
                
                # Update overall status
                if latest_record.status == ComplianceStatus.NON_COMPLIANT:
                    report["overall_status"] = ComplianceStatus.NON_COMPLIANT.value
                elif latest_record.status == ComplianceStatus.REQUIRES_ACTION and report["overall_status"] == ComplianceStatus.COMPLIANT.value:
                    report["overall_status"] = ComplianceStatus.REQUIRES_ACTION.value
        
        return report
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an audit event for compliance tracking."""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details
        }
        
        self.audit_trail.append(event)
        
        # Keep only recent audit events (last 10,000)
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-10000:]
    
    def export_compliance_data(self, file_path: str, format: str = "json"):
        """Export compliance data for external auditing."""
        data = {
            "compliance_records": [
                {
                    "record_id": r.record_id,
                    "regulation": r.regulation.value,
                    "assessment_date": r.assessment_date.isoformat(),
                    "status": r.status.value,
                    "score": r.score,
                    "requirements_met": r.requirements_met,
                    "requirements_failed": r.requirements_failed,
                    "recommendations": r.recommendations
                }
                for r in self.compliance_records
            ],
            "data_subject_requests": [
                {
                    "request_id": r.request_id,
                    "request_type": r.request_type,
                    "regulation": r.regulation.value,
                    "submission_date": r.submission_date.isoformat(),
                    "status": r.status,
                    "response_deadline": r.response_deadline.isoformat()
                }
                for r in self.data_subject_requests
            ],
            "audit_trail": self.audit_trail[-1000:]  # Last 1000 events
        }
        
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Compliance data exported to {file_path}")


def create_compliance_engine(config: FederatedConfig) -> ComplianceEngine:
    """Create compliance engine with configuration."""
    return ComplianceEngine(config)


def assess_system_compliance(
    config: FederatedConfig,
    regulations: List[PrivacyRegulation]
) -> Dict[PrivacyRegulation, ComplianceRecord]:
    """Assess system compliance with multiple regulations."""
    compliance_engine = create_compliance_engine(config)
    
    # System configuration for assessment
    system_config = {
        "consent_mechanism_enabled": True,
        "data_minimization_enabled": True,
        "encryption_enabled": True,
        "breach_notification_system": True,
        "dpo_appointed": True,
        "privacy_config": {
            "epsilon": config.privacy.epsilon,
            "delta": config.privacy.delta
        }
    }
    
    results = {}
    
    for regulation in regulations:
        compliance_engine.enable_regulation(regulation)
        record = compliance_engine.assess_compliance(regulation, system_config)
        results[regulation] = record
    
    return results