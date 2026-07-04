"""
Text locator module for OpenOperator.

Provides spatial OCR mapping to detect screen coordinates of UI elements.
Upgraded to handle multi-word phrases, OCR token merging/splitting, and 
spacing differences using a sliding window aggregator and standard library fuzzy matching.
Includes built-in Retry Strategy for self-healing UI perception.
"""

import difflib
import io
import logging
import re
import time
from typing import Any, Callable, Dict, List, Tuple

import pytesseract
from PIL import Image, UnidentifiedImageError
from PIL import ImageOps
from pydantic import BaseModel

from openoperator.core.models import BoundingBox, UITarget
from openoperator.perception import fuzzy

logger = logging.getLogger(__name__)


class OcrToken(BaseModel):
    """Internal model representing a single localized word from OCR."""
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float


class TextLocatorEngine:
    """
    Engine responsible for finding spatial coordinates of text on the screen.
    Features robust text normalization and N-gram sliding windows to counteract 
    OCR tokenization errors without requiring external fuzzy matching libraries.
    """

    def find_text_targets_with_retry(
        self, 
        image_provider: Callable[[], bytes], 
        search_text: str, 
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        exact_match: bool = False,
        fuzzy_threshold: float = 0.85
    ) -> List[UITarget]:
        """
        Self-healing UI locator. Retries searching if the element is not found
        due to UI load times or transitory screen states.

        Args:
            image_provider (Callable): Function that returns the latest screen capture bytes.
            search_text (str): The target phrase to locate.
            max_attempts (int): Maximum number of OCR attempts.
            retry_delay (float): Seconds to wait between attempts.
            exact_match (bool): If True, requires exact string match.
            fuzzy_threshold (float): Minimum difflib ratio to accept a fuzzy match.
            
        Returns:
            List[UITarget]: Detected targets, or empty list if ultimately not found.
        """
        for attempt in range(max_attempts):
            logger.info(f"Vision search attempt {attempt + 1}/{max_attempts} for '{search_text}'")
            
            # Capture latest screen state dynamically for this attempt
            current_image_bytes = image_provider()
            
            targets = self.find_text_targets(
                image_bytes=current_image_bytes,
                search_text=search_text,
                exact_match=exact_match,
                fuzzy_threshold=fuzzy_threshold,
                dark_mode=dark_mode,
                crop_box=crop_box,
            )
            
            if targets:
                logger.info(f"Target '{search_text}' found on attempt {attempt + 1}.")
                return targets
            
            if attempt < max_attempts - 1:
                logger.warning(f"Target '{search_text}' not found. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
        logger.error(f"Failed to locate '{search_text}' after {max_attempts} attempts.")
        return []

    def find_text_targets(
        self, 
        image_bytes: bytes, 
        search_text: str, 
        exact_match: bool = False,
        fuzzy_threshold: float = 0.85,
        dark_mode: bool = False,
        crop_box: Tuple[int, int, int, int] | None = None,
    ) -> List[UITarget]:
        """
        Locates multi-word text targets on the screen.

        Args:
            image_bytes (bytes): The raw image data.
            search_text (str): The target phrase to locate (e.g., "New chat").
            exact_match (bool): If True, requires exact string match (case-sensitive, exact spaces).
            fuzzy_threshold (float): Minimum difflib ratio (0.0-1.0) to accept a fuzzy match.

        Returns:
            List[UITarget]: Detected targets, sorted by highest confidence.
        """
        if not image_bytes or not search_text.strip():
            logger.warning("Empty image or search text provided. Returning no targets.")
            return []

        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode not in ("RGB", "L", "RGBA"):
                image = image.convert("RGB")

            # Optional cropping to limit OCR region and improve speed/accuracy.
            # crop_box must be (left, top, width, height)
            if crop_box:
                left, top, width, height = crop_box
                right = left + width
                bottom = top + height
                image = image.crop((left, top, right, bottom))

            # Dark mode preprocessing: invert image to improve OCR on inverted UIs
            if dark_mode:
                try:
                    image = ImageOps.invert(image.convert("RGB"))
                except Exception:
                    # If invert fails, continue with original image
                    logger.warning("Dark mode image preprocessing failed; proceeding without inversion.")
        except UnidentifiedImageError:
            logger.error("Failed to parse image bytes.")
            return []

        logger.debug(f"Running spatial OCR for target: '{search_text}'")
        raw_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        line_groups = self._group_words_by_line(raw_data)
        
        # Calculate dynamic window sizes. Tesseract might merge a 2-word phrase into 1 token, 
        # or split a 2-word phrase into 3 tokens.
        target_word_count = len(search_text.split())
        window_sizes = [
            max(1, target_word_count - 1), 
            target_word_count, 
            target_word_count + 1,
            target_word_count + 2
        ]
        
        targets: List[UITarget] = []
        norm_search = self._normalize_text(search_text)

        for line_key, tokens in line_groups.items():
            for window_size in window_sizes:
                if len(tokens) < window_size:
                    continue
                    
                for i in range(len(tokens) - window_size + 1):
                    window_tokens = tokens[i : i + window_size]
                    phrase = " ".join(t.text for t in window_tokens)
                    
                    is_match = False
                    match_score = 0.0

                    if exact_match:
                        if search_text == phrase:
                            is_match = True
                            match_score = 1.0
                    else:
                        norm_phrase = self._normalize_text(phrase)
                        
                        # 1. Exact normalized match (catches spacing/casing/merged word issues instantly)
                        if norm_search == norm_phrase:
                            is_match = True
                            match_score = 1.0
                        # 2. Substring match for tightly packed UI tokens
                        elif norm_search in norm_phrase:
                            is_match = True
                            match_score = 0.95
                        # 3. Fallback to difflib for actual OCR misreadings (e.g., "New cnat")
                        else:
                            # Use the more robust fuzzy matching utilities when available
                            try:
                                ratio = fuzzy.fuzzy_match_score(norm_search, norm_phrase)
                            except Exception:
                                # Fallback to difflib if fuzzy utilities fail
                                ratio = difflib.SequenceMatcher(None, norm_search, norm_phrase).ratio()

                            if ratio >= fuzzy_threshold:
                                is_match = True
                                match_score = ratio

                    if is_match:
                        box = self._merge_bounding_boxes(window_tokens)
                        avg_ocr_conf = sum(t.conf for t in window_tokens) / len(window_tokens)
                        
                        # Blend string similarity ratio with Tesseract's confidence score
                        blended_conf = (match_score * 0.7) + ((avg_ocr_conf / 100.0) * 0.3)
                        
                        targets.append(
                            UITarget(
                                text=phrase,
                                box=box,
                                center_x=box.x + (box.width // 2),
                                center_y=box.y + (box.height // 2),
                                confidence=round(blended_conf * 100, 2)
                            )
                        )

        # Sort by highest confidence
        targets.sort(key=lambda t: t.confidence, reverse=True)
        return self._deduplicate_targets(targets)

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes text by lowercasing and removing all non-alphanumeric characters.
        This entirely eliminates spacing, punctuation, and token-merge errors.
        """
        return re.sub(r'[\W_]+', '', text.lower())

    def _group_words_by_line(self, raw_data: Dict[str, Any]) -> Dict[Tuple[int, int, int], List[OcrToken]]:
        """
        Groups raw Tesseract output into discrete lines based on structural metadata.
        """
        lines: Dict[Tuple[int, int, int], List[OcrToken]] = {}
        
        for i in range(len(raw_data["text"])):
            text = raw_data["text"][i].strip()
            conf = float(raw_data["conf"][i])
            
            # Filter empty strings and Tesseract's negative confidence metadata artifacts
            if not text or conf < 0:
                continue
                
            block_num = raw_data["block_num"][i]
            par_num = raw_data["par_num"][i]
            line_num = raw_data["line_num"][i]
            
            key = (block_num, par_num, line_num)
            if key not in lines:
                lines[key] = []
                
            lines[key].append(
                OcrToken(
                    text=text,
                    left=raw_data["left"][i],
                    top=raw_data["top"][i],
                    width=raw_data["width"][i],
                    height=raw_data["height"][i],
                    conf=conf
                )
            )
            
        return lines

    def _merge_bounding_boxes(self, tokens: List[OcrToken]) -> BoundingBox:
        """Calculates the minimum enclosing bounding box for a sequence of tokens."""
        x_coords = [t.left for t in tokens]
        y_coords = [t.top for t in tokens]
        max_x_coords = [t.left + t.width for t in tokens]
        max_y_coords = [t.top + t.height for t in tokens]
        
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(max_x_coords)
        max_y = max(max_y_coords)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )

    def _deduplicate_targets(self, targets: List[UITarget]) -> List[UITarget]:
        """
        Removes redundant targets resulting from multi-window overlap.
        Keeps the target with the highest confidence score.
        """
        unique_targets: List[UITarget] = []
        
        for target in targets:
            is_duplicate = False
            for unique in unique_targets:
                dist_x = abs(target.center_x - unique.center_x)
                dist_y = abs(target.center_y - unique.center_y)
                
                # If centers are within the bounds of each other, they are duplicates
                if dist_x < max(target.box.width, unique.box.width) and \
                   dist_y < max(target.box.height, unique.box.height):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_targets.append(target)
                
        return unique_targets