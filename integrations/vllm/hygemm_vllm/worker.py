class AuditWorkerExtension:
    def hygemm_model_audit(self):
        from .quantization import model_audit
        return model_audit(self.model_runner.model)
