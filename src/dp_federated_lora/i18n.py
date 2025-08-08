"""
Internationalization (i18n) support for DP-Federated LoRA Lab.

Provides multi-language support for global deployment with GDPR, CCPA,
and PDPA compliance messaging.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class LocalizedMessages:
    """Container for localized messages."""
    privacy_consent: str
    data_processing_notice: str
    federated_training_info: str
    error_messages: Dict[str, str]
    system_messages: Dict[str, str]


class InternationalizationManager:
    """Manages internationalization for global federated learning."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.messages = self._load_default_messages()
        
    def _load_default_messages(self) -> Dict[SupportedLanguage, LocalizedMessages]:
        """Load default localized messages."""
        return {
            SupportedLanguage.ENGLISH: LocalizedMessages(
                privacy_consent="By participating in federated learning, your data remains on your device while contributing to model improvement with differential privacy guarantees (ε-δ).",
                data_processing_notice="Data Processing Notice: This system implements differential privacy (DP-SGD) to protect individual privacy during federated learning. Your local model updates are noised before aggregation.",
                federated_training_info="Quantum-Enhanced Federated Learning: Your device participates in collaborative model training without sharing raw data. Privacy budget: ε={epsilon}, δ={delta}",
                error_messages={
                    "connection_failed": "Failed to connect to federated server. Please check your network connection.",
                    "privacy_budget_exceeded": "Privacy budget exceeded. Training stopped to maintain privacy guarantees.",
                    "authentication_failed": "Authentication failed. Please verify your client credentials.",
                    "model_sync_failed": "Model synchronization failed. Retrying with exponential backoff."
                },
                system_messages={
                    "training_started": "Federated training session started",
                    "training_completed": "Training completed successfully. Privacy spent: ε={epsilon}",
                    "client_joined": "Client {client_id} joined the federation",
                    "round_completed": "Training round {round} completed"
                }
            ),
            SupportedLanguage.SPANISH: LocalizedMessages(
                privacy_consent="Al participar en el aprendizaje federado, sus datos permanecen en su dispositivo mientras contribuyen a la mejora del modelo con garantías de privacidad diferencial (ε-δ).",
                data_processing_notice="Aviso de Procesamiento de Datos: Este sistema implementa privacidad diferencial (DP-SGD) para proteger la privacidad individual durante el aprendizaje federado.",
                federated_training_info="Aprendizaje Federado Cuántico: Su dispositivo participa en entrenamiento colaborativo sin compartir datos brutos. Presupuesto de privacidad: ε={epsilon}, δ={delta}",
                error_messages={
                    "connection_failed": "Error al conectar con el servidor federado. Verifique su conexión de red.",
                    "privacy_budget_exceeded": "Presupuesto de privacidad excedido. Entrenamiento detenido para mantener garantías de privacidad.",
                    "authentication_failed": "Autenticación fallida. Verifique sus credenciales de cliente.",
                    "model_sync_failed": "Sincronización del modelo fallida. Reintentando con retroceso exponencial."
                },
                system_messages={
                    "training_started": "Sesión de entrenamiento federado iniciada",
                    "training_completed": "Entrenamiento completado exitosamente. Privacidad gastada: ε={epsilon}",
                    "client_joined": "Cliente {client_id} se unió a la federación",
                    "round_completed": "Ronda de entrenamiento {round} completada"
                }
            ),
            SupportedLanguage.FRENCH: LocalizedMessages(
                privacy_consent="En participant à l'apprentissage fédéré, vos données restent sur votre appareil tout en contribuant à l'amélioration du modèle avec des garanties de confidentialité différentielle (ε-δ).",
                data_processing_notice="Avis de Traitement des Données: Ce système implémente la confidentialité différentielle (DP-SGD) pour protéger la vie privée individuelle pendant l'apprentissage fédéré.",
                federated_training_info="Apprentissage Fédéré Quantique: Votre appareil participe à l'entraînement collaboratif sans partager de données brutes. Budget de confidentialité: ε={epsilon}, δ={delta}",
                error_messages={
                    "connection_failed": "Échec de connexion au serveur fédéré. Vérifiez votre connexion réseau.",
                    "privacy_budget_exceeded": "Budget de confidentialité dépassé. Entraînement arrêté pour maintenir les garanties de confidentialité.",
                    "authentication_failed": "Échec d'authentification. Vérifiez vos identifiants client.",
                    "model_sync_failed": "Échec de synchronisation du modèle. Nouvelle tentative avec backoff exponentiel."
                },
                system_messages={
                    "training_started": "Session d'entraînement fédéré démarrée",
                    "training_completed": "Entraînement terminé avec succès. Confidentialité dépensée: ε={epsilon}",
                    "client_joined": "Client {client_id} a rejoint la fédération",
                    "round_completed": "Round d'entraînement {round} terminé"
                }
            ),
            SupportedLanguage.GERMAN: LocalizedMessages(
                privacy_consent="Durch die Teilnahme am föderierten Lernen verbleiben Ihre Daten auf Ihrem Gerät und tragen zur Modellverbesserung mit differenziellen Datenschutzgarantien (ε-δ) bei.",
                data_processing_notice="Datenverarbeitungshinweis: Dieses System implementiert differenziellen Datenschutz (DP-SGD) zum Schutz der individuellen Privatsphäre während des föderierten Lernens.",
                federated_training_info="Quanten-verstärktes Föderiertes Lernen: Ihr Gerät nimmt am kollaborativen Modelltraining teil, ohne Rohdaten zu teilen. Datenschutz-Budget: ε={epsilon}, δ={delta}",
                error_messages={
                    "connection_failed": "Verbindung zum föderierten Server fehlgeschlagen. Überprüfen Sie Ihre Netzwerkverbindung.",
                    "privacy_budget_exceeded": "Datenschutz-Budget überschritten. Training gestoppt, um Datenschutzgarantien aufrechtzuerhalten.",
                    "authentication_failed": "Authentifizierung fehlgeschlagen. Überprüfen Sie Ihre Client-Anmeldedaten.",
                    "model_sync_failed": "Modellsynchronisation fehlgeschlagen. Wiederholung mit exponentieller Verzögerung."
                },
                system_messages={
                    "training_started": "Föderierte Trainingssitzung gestartet",
                    "training_completed": "Training erfolgreich abgeschlossen. Datenschutz verbraucht: ε={epsilon}",
                    "client_joined": "Client {client_id} ist der Föderation beigetreten",
                    "round_completed": "Trainingsrunde {round} abgeschlossen"
                }
            ),
            SupportedLanguage.JAPANESE: LocalizedMessages(
                privacy_consent="連合学習に参加することで、お客様のデータはデバイス上に残りながら、差分プライバシー保証（ε-δ）でモデル改善に貢献します。",
                data_processing_notice="データ処理に関する通知：このシステムは、連合学習中の個人プライバシーを保護するために差分プライバシー（DP-SGD）を実装しています。",
                federated_training_info="量子強化連合学習：お客様のデバイスは生データを共有することなく、協調的なモデルトレーニングに参加します。プライバシー予算：ε={epsilon}、δ={delta}",
                error_messages={
                    "connection_failed": "連合サーバーへの接続に失敗しました。ネットワーク接続を確認してください。",
                    "privacy_budget_exceeded": "プライバシー予算を超過しました。プライバシー保証を維持するためトレーニングを停止しました。",
                    "authentication_failed": "認証に失敗しました。クライアント認証情報を確認してください。",
                    "model_sync_failed": "モデル同期に失敗しました。指数バックオフで再試行中です。"
                },
                system_messages={
                    "training_started": "連合トレーニングセッションが開始されました",
                    "training_completed": "トレーニングが正常に完了しました。使用されたプライバシー：ε={epsilon}",
                    "client_joined": "クライアント{client_id}が連合に参加しました",
                    "round_completed": "トレーニングラウンド{round}が完了しました"
                }
            ),
            SupportedLanguage.CHINESE: LocalizedMessages(
                privacy_consent="通过参与联邦学习，您的数据保留在您的设备上，同时通过差分隐私保证（ε-δ）为模型改进做出贡献。",
                data_processing_notice="数据处理通知：该系统实施差分隐私（DP-SGD）以在联邦学习期间保护个人隐私。",
                federated_training_info="量子增强联邦学习：您的设备参与协作模型训练，无需共享原始数据。隐私预算：ε={epsilon}，δ={delta}",
                error_messages={
                    "connection_failed": "连接联邦服务器失败。请检查您的网络连接。",
                    "privacy_budget_exceeded": "隐私预算超出。已停止训练以维护隐私保证。",
                    "authentication_failed": "身份验证失败。请验证您的客户端凭据。",
                    "model_sync_failed": "模型同步失败。正在使用指数退避重试。"
                },
                system_messages={
                    "training_started": "联邦训练会话已开始",
                    "training_completed": "训练成功完成。隐私消耗：ε={epsilon}",
                    "client_joined": "客户端{client_id}已加入联邦",
                    "round_completed": "训练轮次{round}已完成"
                }
            )
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set the current language."""
        self.current_language = language
        logger.info(f"Language set to {language.value}")
    
    def get_message(self, key: str, category: str = "system_messages", **kwargs) -> str:
        """Get a localized message with optional formatting."""
        try:
            messages = self.messages[self.current_language]
            
            if category == "error_messages":
                message = messages.error_messages.get(key)
            elif category == "system_messages":
                message = messages.system_messages.get(key)
            elif category == "privacy_consent":
                message = messages.privacy_consent
            elif category == "data_processing_notice":
                message = messages.data_processing_notice
            elif category == "federated_training_info":
                message = messages.federated_training_info
            else:
                message = None
                
            if message and kwargs:
                return message.format(**kwargs)
            return message or f"[Missing translation: {key}]"
            
        except Exception as e:
            logger.error(f"Failed to get localized message: {e}")
            # Fallback to English
            fallback_messages = self.messages[SupportedLanguage.ENGLISH]
            if category == "error_messages":
                return fallback_messages.error_messages.get(key, f"[Translation error: {key}]")
            elif category == "system_messages":
                return fallback_messages.system_messages.get(key, f"[Translation error: {key}]")
            return f"[Translation error: {key}]"
    
    def get_privacy_consent(self, **kwargs) -> str:
        """Get localized privacy consent message."""
        return self.get_message("", "privacy_consent", **kwargs)
    
    def get_data_processing_notice(self, **kwargs) -> str:
        """Get localized data processing notice."""
        return self.get_message("", "data_processing_notice", **kwargs)
    
    def get_federated_training_info(self, epsilon: float, delta: float) -> str:
        """Get localized federated training information."""
        return self.get_message("", "federated_training_info", epsilon=epsilon, delta=delta)
    
    def get_compliance_text(self, regulation: str = "GDPR") -> str:
        """Get compliance-specific text."""
        compliance_texts = {
            "GDPR": {
                SupportedLanguage.ENGLISH: "This system complies with GDPR requirements for data protection and privacy.",
                SupportedLanguage.SPANISH: "Este sistema cumple con los requisitos del RGPD para la protección de datos y privacidad.",
                SupportedLanguage.FRENCH: "Ce système respecte les exigences du RGPD pour la protection des données et de la vie privée.",
                SupportedLanguage.GERMAN: "Dieses System entspricht den DSGVO-Anforderungen für Datenschutz und Privatsphäre.",
                SupportedLanguage.JAPANESE: "このシステムはデータ保護とプライバシーに関するGDPR要件に準拠しています。",
                SupportedLanguage.CHINESE: "该系统符合GDPR数据保护和隐私要求。"
            },
            "CCPA": {
                SupportedLanguage.ENGLISH: "This system complies with CCPA requirements for California consumer privacy.",
                SupportedLanguage.SPANISH: "Este sistema cumple con los requisitos de la CCPA para la privacidad del consumidor de California."
            },
            "PDPA": {
                SupportedLanguage.ENGLISH: "This system complies with PDPA requirements for personal data protection.",
                SupportedLanguage.JAPANESE: "このシステムは個人データ保護に関するPDPA要件に準拠しています。",
                SupportedLanguage.CHINESE: "该系统符合PDPA个人数据保护要求。"
            }
        }
        
        return compliance_texts.get(regulation, {}).get(
            self.current_language, 
            compliance_texts.get(regulation, {}).get(SupportedLanguage.ENGLISH, "Compliance information not available.")
        )


# Global i18n manager instance
i18n_manager = InternationalizationManager()


def get_localized_message(key: str, category: str = "system_messages", **kwargs) -> str:
    """Convenience function to get localized messages."""
    return i18n_manager.get_message(key, category, **kwargs)


def set_global_language(language: SupportedLanguage) -> None:
    """Set the global language for the application."""
    i18n_manager.set_language(language)