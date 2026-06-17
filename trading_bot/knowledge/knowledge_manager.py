import os
import logging

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Singleton pattern per caricare e mantenere in memoria RAM 
    tutta la conoscenza testuale dell'agente al momento del boot.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeManager, cls).__new__(cls)
            cls._instance._knowledge_cache = {}
            cls._instance._preload_knowledge()
        return cls._instance

    def _preload_knowledge(self):
        """Legge i file dalla cartella 'knowledge' (situata nella root del progetto)"""
        logger.info("[SYSTEM] Inizializzazione Knowledge Base in corso...")
        
        # MAGIA DEI PATH: Calcola dinamicamente il percorso assoluto alla cartella 'knowledge'
        # __file__ è questo file (trading_bot/knowledge_manager.py)
        # os.path.dirname(__file__) è la cartella 'trading_bot'
        # os.path.dirname(...) ancora, è la ROOT del progetto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = os.path.join(project_root, "knowledge")
        
        # LISTA ESATTA DEI TUOI FILE
        # Assicurati che i nomi corrispondano esattamente ai file creati
        files_to_load = [
            "the_intelligent_investor.txt",
            "quantities.txt"
        ]
        
        if not os.path.exists(base_dir):
            logger.warning(f"[WARNING] Cartella '{base_dir}' non trovata. Creata automaticamente.")
            os.makedirs(base_dir, exist_ok=True)

        for filename in files_to_load:
            filepath = os.path.join(base_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self._knowledge_cache[filename] = f.read()
                logger.info(f"[OK] Caricato in memoria: {filename}")
            except FileNotFoundError:
                logger.error(f"[ERROR] File '{filename}' non trovato nel percorso: {filepath}")
                self._knowledge_cache[filename] = ""

    def get_knowledge(self, filename: str) -> str:
        """Restituisce il contenuto del file richiesto dalla cache."""
        return self._knowledge_cache.get(filename, "")

# Istanzia il Singleton
knowledge_base = KnowledgeManager()