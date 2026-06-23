import numpy as np
import json
import os


class TreatmentBandit:
    """
    Thompson Sampling bandit for treatment recommendation.
    Tracks a Beta(alpha, beta) distribution per treatment, per condition category.
    """

    def __init__(self, treatments_by_category=None, state_path=None):
        self.state_path = state_path

        if treatments_by_category is None:
            treatments_by_category = self._default_treatments()
        self.treatments_by_category = treatments_by_category

        # Load existing state if a path is given and exists, else initialize fresh
        if state_path and os.path.exists(state_path):
            self.load_state(state_path)
        else:
            self.state = self._init_state(treatments_by_category)

    @staticmethod
    def _default_treatments():
        return {
            'Respiratory': ['Inhaled Bronchodilator', 'Antibiotic Course', 'Rest + Hydration Protocol'],
            'Gastrointestinal': ['Dietary Modification', 'Antacid/PPI Therapy', 'Antibiotic Course'],
            'Cardiovascular': ['ACE Inhibitor', 'Lifestyle + Statin Therapy', 'Beta Blocker'],
            'Neurological': ['Neurology Referral + MRI', 'Symptom Management Meds', 'Physical Therapy'],
            'Musculoskeletal': ['NSAID + Rest', 'Physical Therapy', 'Corticosteroid Injection'],
            'Endocrine': ['Hormone Therapy', 'Dietary + Insulin Management', 'Endocrinology Referral'],
            'Infectious Disease': ['Antibiotic Course', 'Antiviral Therapy', 'Supportive Care + Isolation'],
            'Cancer': ['Oncology Referral', 'Chemotherapy Protocol', 'Surgical Consultation'],
            'Mental Health': ['Psychotherapy Referral', 'SSRI Medication', 'Combined Therapy + Medication'],
            'Renal/Urinary': ['Antibiotic Course', 'Increased Hydration + Monitoring', 'Nephrology Referral'],
            'Dermatological': ['Topical Steroid', 'Antihistamine', 'Dermatology Referral'],
            'Genetic/Congenital': ['Specialist Referral', 'Supportive Care Plan', 'Genetic Counseling'],
        }

    @staticmethod
    def _init_state(treatments_by_category):
        state = {}
        for category, treatments in treatments_by_category.items():
            state[category] = {t: {'alpha': 1, 'beta': 1} for t in treatments}
        return state

    def recommend(self, category):
        """Sample from each treatment's Beta distribution, return the best one + all samples."""
        if category not in self.state:
            raise ValueError(f"Unknown category: {category}")

        treatments = self.state[category]
        samples = {
            t: np.random.beta(p['alpha'], p['beta'])
            for t, p in treatments.items()
        }
        best_treatment = max(samples, key=samples.get)
        return best_treatment, samples

    def update(self, category, treatment, doctor_rating):
        """doctor_rating: 1-5. >=4 = success (alpha+=1), <4 = failure (beta+=1)."""
        if category not in self.state or treatment not in self.state[category]:
            raise ValueError(f"Unknown category/treatment: {category}/{treatment}")

        if doctor_rating >= 4:
            self.state[category][treatment]['alpha'] += 1
        else:
            self.state[category][treatment]['beta'] += 1

        if self.state_path:
            self.save_state(self.state_path)

        return self.state[category][treatment]

    def get_stats(self, category):
        """Return current alpha/beta + estimated success rate per treatment for a category."""
        stats = {}
        for t, p in self.state[category].items():
            alpha, beta = p['alpha'], p['beta']
            stats[t] = {
                'alpha': alpha,
                'beta': beta,
                'estimated_success_rate': round(alpha / (alpha + beta), 3),
                'total_trials': alpha + beta - 2  # subtract the prior of 1,1
            }
        return stats

    def save_state(self, path):
        with open(path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_state(self, path):
        with open(path, 'r') as f:
            self.state = json.load(f)


if __name__ == "__main__":
    # Quick smoke test when running this file directly
    bandit = TreatmentBandit()
    treatment, samples = bandit.recommend('Respiratory')
    print("Recommended:", treatment)
    print("Samples:", samples)

    bandit.update('Respiratory', treatment, doctor_rating=5)
    print(bandit.get_stats('Respiratory'))
