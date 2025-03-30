from typing import Dict, List


class MBTIData:
    """
    Data class for MBTI personality types and movie preferences.
    """

    @staticmethod
    def get_mbti_groups() -> Dict[str, List[str]]:
        """
        Get MBTI types grouped by category.

        Returns:
            Dictionary mapping group names to lists of MBTI types
        """
        return {
            "Analysts (NT)": ["INTJ", "INTP", "ENTJ", "ENTP"],
            "Diplomats (NF)": ["INFJ", "INFP", "ENFJ", "ENFP"],
            "Sentinels (SJ)": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
            "Explorers (SP)": ["ISTP", "ISFP", "ESTP", "ESFP"],
        }

    @staticmethod
    def get_mbti_descriptions() -> Dict[str, str]:
        """
        Get descriptions for each MBTI type.

        Returns:
            Dictionary mapping MBTI types to descriptions
        """
        return {
            "INTJ": "Architect - Imaginative and strategic thinkers, with a plan for everything.",
            "INTP": "Logician - Innovative inventors with an unquenchable thirst for knowledge.",
            "ENTJ": "Commander - Bold, imaginative and strong-willed leaders, always finding a way.",
            "ENTP": "Debater - Smart and curious thinkers who cannot resist an intellectual challenge.",
            "INFJ": "Advocate - Quiet and mystical, yet very inspiring and tireless idealists.",
            "INFP": "Mediator - Poetic, kind and altruistic people, always eager to help a good cause.",
            "ENFJ": "Protagonist - Charismatic and inspiring leaders, able to mesmerize their listeners.",
            "ENFP": "Campaigner - Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.",
            "ISTJ": "Logistician - Practical and fact-minded individuals, whose reliability cannot be doubted.",
            "ISFJ": "Defender - Very dedicated and warm protectors, always ready to defend their loved ones.",
            "ESTJ": "Executive - Excellent administrators, unsurpassed at managing things or people.",
            "ESFJ": "Consul - Extraordinarily caring, social and popular people, always eager to help.",
            "ISTP": "Virtuoso - Bold and practical experimenters, masters of all kinds of tools.",
            "ISFP": "Adventurer - Flexible and charming artists, always ready to explore and experience something new.",
            "ESTP": "Entrepreneur - Smart, energetic and very perceptive people, who truly enjoy living on the edge.",
            "ESFP": "Entertainer - Spontaneous, energetic and enthusiastic entertainers who truly enjoy living in the moment.",
        }

    @staticmethod
    def get_mbti_colors() -> Dict[str, str]:
        """
        Get color codes associated with each MBTI type.

        Returns:
            Dictionary mapping MBTI types to hex color codes
        """
        return {
            # Analysts (NT) - Blues
            "INTJ": "#1A5276",
            "INTP": "#2980B9",
            "ENTJ": "#154360",
            "ENTP": "#3498DB",
            # Diplomats (NF) - Greens
            "INFJ": "#0B5345",
            "INFP": "#27AE60",
            "ENFJ": "#145A32",
            "ENFP": "#2ECC71",
            # Sentinels (SJ) - Purples
            "ISTJ": "#4A235A",
            "ISFJ": "#8E44AD",
            "ESTJ": "#6C3483",
            "ESFJ": "#9B59B6",
            # Explorers (SP) - Reds/Oranges
            "ISTP": "#922B21",
            "ISFP": "#E74C3C",
            "ESTP": "#A04000",
            "ESFP": "#F39C12",
        }
