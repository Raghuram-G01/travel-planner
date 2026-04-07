"""
JARVIS Travel Planner — AI Trip Generator
Rule-based filtering + Ranking Algorithm + Budget Optimizer
No external LLM required — smart local intelligence
"""

import json
import math
from typing import List, Dict, Any, Optional, Tuple
from rag.pipeline import get_store, get_raw_data


# ─── Ranking Algorithm ────────────────────────────────────────────────────────
def rank_places(places: List[Dict], preferences: List[str], budget_per_day: float) -> List[Dict]:
    """
    Score = 0.4 × rating + 0.3 × popularity_norm + 0.2 × preference_match + 0.1 × cost_efficiency
    """
    scored = []
    for place in places:
        rating_score = place.get("rating", 3.0) / 5.0
        popularity_norm = place.get("popularity", 50) / 100.0
        
        # Preference match score
        place_tags = set(place.get("tags", []))
        pref_set = set(preferences)
        overlap = len(place_tags & pref_set)
        pref_score = min(overlap / max(len(pref_set), 1), 1.0)

        # Cost efficiency (lower cost relative to day budget = more efficient)
        cost = place.get("cost", 0)
        if cost == 0:
            cost_eff = 1.0
        else:
            cost_eff = max(0, 1.0 - cost / (budget_per_day * 0.5))

        total = (
            0.40 * rating_score +
            0.30 * popularity_norm +
            0.20 * pref_score +
            0.10 * cost_eff
        )
        scored.append({**place, "_score": round(total, 4)})

    return sorted(scored, key=lambda x: x["_score"], reverse=True)


def rank_hotels(hotels: List[Dict], budget_per_night: float) -> List[Dict]:
    """Rank hotels by value — filter by budget then sort by rating"""
    eligible = [h for h in hotels if h.get("price_per_night", 999999) <= budget_per_night * 1.2]
    if not eligible:
        eligible = sorted(hotels, key=lambda h: h.get("price_per_night", 0))[:2]
    return sorted(eligible, key=lambda h: h.get("rating", 0), reverse=True)


# ─── Budget Optimizer ─────────────────────────────────────────────────────────
def optimize_budget(total_budget: float, days: int) -> Dict[str, float]:
    """
    Greedy allocation of budget across categories
    Returns per-day and total amounts per category
    """
    # Typical Indian travel budget ratios
    ratios = {
        "hotel":     0.35,
        "food":      0.20,
        "transport": 0.15,
        "activities":0.20,
        "shopping":  0.07,
        "emergency": 0.03,
    }
    allocation = {}
    for cat, ratio in ratios.items():
        allocation[cat] = round(total_budget * ratio, 0)

    # Per-day breakdown
    per_day = {
        "hotel_per_night": round(allocation["hotel"] / days, 0),
        "food_per_day":    round(allocation["food"] / days, 0),
        "transport_per_day": round(allocation["transport"] / days, 0),
        "activities_per_day": round(allocation["activities"] / days, 0),
    }
    return {"total": allocation, "per_day": per_day}


# ─── Rule-based Destination Finder ───────────────────────────────────────────
def find_destination(raw_data: Dict, destination_query: str) -> Optional[Dict]:
    """Find destination by fuzzy name match"""
    query_lower = destination_query.lower().strip()
    for dest in raw_data["destinations"]:
        if (dest["id"].lower() in query_lower or
                dest["name"].lower() in query_lower or
                query_lower in dest["name"].lower() or
                dest["state"].lower() in query_lower):
            return dest
    return None


# ─── Day planner ─────────────────────────────────────────────────────────────
def build_day_plan(
    places: List[Dict],
    day_num: int,
    budget_per_day: float,
    food_spots: List[Dict],
    transport: List[Dict],
) -> Dict:
    """Build a single day's itinerary"""
    day_budget = budget_per_day
    day_activities = []
    spent = 0
    hours_used = 0

    for place in places:
        cost = place.get("cost", 0)
        duration = place.get("duration_hours", 2)
        if hours_used + duration > 10:  # max 10 hours activities
            break
        if spent + cost > day_budget * 0.7:  # leave room for food+transport
            if cost > 0:
                continue
        day_activities.append(place)
        spent += cost
        hours_used += duration

    # Pick food spot
    food = food_spots[(day_num - 1) % len(food_spots)] if food_spots else None
    transport_choice = transport[0] if transport else None

    return {
        "day": day_num,
        "title": f"Day {day_num} — {', '.join(p['name'] for p in day_activities[:2])}",
        "activities": day_activities,
        "meals": food,
        "transport": transport_choice,
        "estimated_cost": round(spent + (food["avg_cost_per_person"] * 2 if food else 400)),
    }


# ─── Main Trip Generator ──────────────────────────────────────────────────────
def generate_trip(
    destination: str,
    days: int,
    budget: float,
    preferences: List[str],
) -> Dict[str, Any]:
    """
    Main AI trip generation function
    1. Find destination in knowledge base
    2. Rank places by preferences + budget
    3. Optimize budget allocation
    4. Build day-wise itinerary
    5. Augment with RAG context
    """
    raw_data = get_raw_data()
    store = get_store()

    # 1. Find destination
    dest = find_destination(raw_data, destination)
    if not dest:
        # Try RAG fallback
        rag_results = store.search(destination, top_k=3)
        dest_names = list({r["destination"] for r in rag_results if r["destination"] != "general"})
        if dest_names:
            dest = find_destination(raw_data, dest_names[0])
        if not dest:
            return {
                "error": True,
                "message": f"Destination '{destination}' not found. Available: Goa, Manali, Jaipur, Kerala, Udaipur, Shimla",
                "suggestions": ["Goa", "Manali", "Jaipur", "Kerala", "Udaipur", "Shimla"]
            }

    # 2. Budget optimization
    budget_plan = optimize_budget(budget, days)
    hotel_budget_per_night = budget_plan["per_day"]["hotel_per_night"]

    # 3. Preference-based ranking
    mapped_prefs = map_preferences(preferences)
    ranked_places = rank_places(dest["places"], mapped_prefs, budget / days)

    # 4. Filter places by budget (entry cost must fit daily activities budget)
    max_activity_cost = budget_plan["per_day"]["activities_per_day"]
    affordable_places = [
        p for p in ranked_places
        if p["cost"] <= max_activity_cost
    ]
    if not affordable_places:
        affordable_places = ranked_places  # fallback - include all

    # 5. Select hotel
    hotels = rank_hotels(dest["hotels"], hotel_budget_per_night)
    chosen_hotel = hotels[0] if hotels else dest["hotels"][0]

    # 6. Build day-wise itinerary
    places_pool = affordable_places.copy()
    itinerary = []
    for day in range(1, days + 1):
        # Rotate places to avoid repetition
        start_idx = (day - 1) * 3
        day_places = places_pool[start_idx:start_idx + 4] or places_pool[:4]
        day_plan = build_day_plan(
            places=day_places,
            day_num=day,
            budget_per_day=budget / days,
            food_spots=dest.get("food", []),
            transport=dest.get("transport", []),
        )
        itinerary.append(day_plan)

    # 7. RAG enhancement — get contextual tips
    rag_query = f"{destination} travel tips {' '.join(preferences)}"
    rag_docs = store.search(rag_query, top_k=5)
    tips = [
        d["raw"]["tip"]
        for d in rag_docs
        if d["type"] == "tip"
    ][:3]

    # 8. Budget breakdown
    total_hotel = chosen_hotel["price_per_night"] * days
    total_food = budget_plan["total"]["food"]
    total_transport = budget_plan["total"]["transport"]
    total_activities = sum(
        sum(act.get("cost", 0) for act in day["activities"])
        for day in itinerary
    )
    total_estimated = total_hotel + total_food + total_transport + total_activities

    return {
        "destination": dest["name"],
        "state": dest["state"],
        "days": days,
        "total_budget": budget,
        "description": dest["description"],
        "best_season": dest["best_season"],
        "hotel": chosen_hotel,
        "itinerary": itinerary,
        "budget_breakdown": {
            "hotel": round(total_hotel),
            "food": round(total_food),
            "transport": round(total_transport),
            "activities": round(total_activities),
            "shopping_emergency": round(budget_plan["total"]["shopping"] + budget_plan["total"]["emergency"]),
            "total_estimated": round(total_estimated),
            "savings": round(max(0, budget - total_estimated)),
        },
        "tips": tips,
        "recommended_transport": dest.get("transport", [])[:2],
        "must_try_food": dest.get("food", [])[:3],
        "preference_tags": mapped_prefs,
        "rag_context_used": len(rag_docs),
    }


def map_preferences(preferences: List[str]) -> List[str]:
    """Map user-friendly preferences to internal tags"""
    mapping = {
        "adventure": ["adventure", "trekking", "water_sports", "sports"],
        "food": ["food", "culture", "local"],
        "nightlife": ["nightlife", "party", "beach"],
        "beaches": ["beach", "water_sports", "sunset"],
        "history": ["history", "culture", "architecture"],
        "nature": ["nature", "wildlife", "trekking", "peaceful"],
        "shopping": ["shopping", "culture", "local"],
        "romance": ["romance", "sunset", "peaceful", "luxury"],
        "budget": ["budget", "backpacker"],
        "luxury": ["luxury", "fine_dining", "spa"],
    }
    tags = []
    for pref in preferences:
        pref_lower = pref.lower()
        for key, vals in mapping.items():
            if key in pref_lower or pref_lower in key:
                tags.extend(vals)
    return list(set(tags)) if tags else ["adventure", "culture", "food"]
