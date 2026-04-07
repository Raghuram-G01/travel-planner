"""
JARVIS Travel Planner — Conversational Chatbot Engine
Context-aware travel assistant with session memory
"""

import re
from typing import List, Dict, Any, Optional
from rag.pipeline import get_store, get_raw_data
from rag.ai_engine import generate_trip, map_preferences, find_destination


# ─── Session Memory ───────────────────────────────────────────────────────────
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.context: Dict = {
            "destination": None,
            "days": None,
            "budget": None,
            "preferences": [],
            "last_trip": None,
        }

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def update_context(self, updates: Dict):
        self.context.update(updates)

    def is_complete(self) -> bool:
        return all([
            self.context["destination"],
            self.context["days"],
            self.context["budget"],
        ])


# In-memory session store
_sessions: Dict[str, ChatSession] = {}

def get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(session_id)
    return _sessions[session_id]


# ─── NLP Extractors ───────────────────────────────────────────────────────────
DESTINATIONS = ["goa", "manali", "jaipur", "kerala", "udaipur", "shimla", "rajasthan"]
PREFERENCES_KEYWORDS = ["adventure", "food", "nightlife", "beaches", "history", "nature", "shopping", "romance", "budget", "luxury"]

def extract_destination(text: str) -> Optional[str]:
    t = text.lower()
    for dest in DESTINATIONS:
        if dest in t:
            return dest.capitalize()
    # Check for state names
    states = {"rajasthan": "Jaipur", "himachal": "Manali", "kerala": "Kerala", "goa": "Goa"}
    for state, dest in states.items():
        if state in t:
            return dest
    return None

def extract_days(text: str) -> Optional[int]:
    patterns = [
        r"(\d+)\s*(?:day|days|night|nights)",
        r"for\s+(\d+)",
        r"(\d+)\s*-\s*day",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            return int(m.group(1))
    # Word numbers
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    for w, n in words.items():
        if w in text.lower():
            return n
    return None

def extract_budget(text: str) -> Optional[float]:
    patterns = [
        r"₹\s*([\d,]+)",
        r"rs\.?\s*([\d,]+)",
        r"inr\s*([\d,]+)",
        r"([\d,]+)\s*(?:rupees|rs|inr|₹|k(?!\w))",
        r"budget.*?([\d,]+)",
        r"under\s+([\d,]+)",
        r"([\d]+)k\b",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            raw = m.group(1).replace(",", "")
            val = float(raw)
            if val < 1000:  # e.g., "10k" → "10" → 10000
                val *= 1000
            return val
    return None

def extract_preferences(text: str) -> List[str]:
    t = text.lower()
    found = []
    for pref in PREFERENCES_KEYWORDS:
        if pref in t:
            found.append(pref)
    return found


# ─── Response Generator ───────────────────────────────────────────────────────
def generate_chat_response(session_id: str, user_message: str) -> Dict[str, Any]:
    session = get_session(session_id)
    session.add_message("user", user_message)
    store = get_store()
    raw_data = get_raw_data()
    text_lower = user_message.lower()

    # ── Detect intent ────────────────────────────────────────────────────────
    is_plan_request = any(w in text_lower for w in ["plan", "trip", "travel", "visit", "go to", "itinerary", "book"])
    is_info_request = any(w in text_lower for w in ["tell me", "what", "where", "how", "best", "recommend", "suggest", "info", "information"])
    is_modify_request = any(w in text_lower for w in ["change", "modify", "update", "add", "remove", "instead", "more", "less", "cheaper", "budget"])
    is_greeting = any(w in text_lower for w in ["hello", "hi", "hey", "jarvis", "start", "help"])

    # ── Extract entities ─────────────────────────────────────────────────────
    extracted_dest = extract_destination(user_message)
    extracted_days = extract_days(user_message)
    extracted_budget = extract_budget(user_message)
    extracted_prefs = extract_preferences(user_message)

    # Update session context
    if extracted_dest:
        session.update_context({"destination": extracted_dest})
    if extracted_days:
        session.update_context({"days": extracted_days})
    if extracted_budget:
        session.update_context({"budget": extracted_budget})
    if extracted_prefs:
        existing = session.context.get("preferences", [])
        session.update_context({"preferences": list(set(existing + extracted_prefs))})

    ctx = session.context

    # ── Greeting ─────────────────────────────────────────────────────────────
    if is_greeting and not is_plan_request:
        response = {
            "message": (
                "🤖 **JARVIS Travel AI online.** I'm your intelligent travel planning assistant.\n\n"
                "I can help you:\n"
                "• 🗺️ Plan personalized day-by-day itineraries\n"
                "• 💰 Optimize your travel budget\n"
                "• 🏨 Find the best hotels within your budget\n"
                "• 🍽️ Discover local food & experiences\n\n"
                "Try asking: *\"Plan a 3-day Goa trip under ₹15,000 with beaches and adventure\"*"
            ),
            "type": "greeting",
            "suggestions": [
                "Plan a 3-day Goa trip under ₹15k",
                "Best places to visit in Manali",
                "5-day Kerala tour for ₹25,000",
                "Jaipur itinerary with food and history"
            ]
        }
        session.add_message("assistant", response["message"])
        return response

    # ── Modification request ──────────────────────────────────────────────────
    if is_modify_request and session.context.get("last_trip"):
        # Re-generate with updated context
        if extracted_budget:
            session.update_context({"budget": extracted_budget})
        if extracted_days:
            session.update_context({"days": extracted_days})

        if session.is_complete():
            trip = generate_trip(ctx["destination"], ctx["days"], ctx["budget"], ctx["preferences"])
            session.update_context({"last_trip": trip})
            response = {
                "message": f"✅ I've updated your {ctx['destination']} trip plan with the new parameters!",
                "type": "trip_update",
                "trip": trip,
            }
            session.add_message("assistant", response["message"])
            return response

    # ── Full plan generation ──────────────────────────────────────────────────
    if is_plan_request or (extracted_dest and (extracted_days or ctx["days"])):
        if session.is_complete():
            trip = generate_trip(ctx["destination"], ctx["days"], ctx["budget"], ctx["preferences"])
            session.update_context({"last_trip": trip})

            if not trip.get("error"):
                msg = (
                    f"🚀 **JARVIS has generated your {ctx['destination']} itinerary!**\n\n"
                    f"📅 **{ctx['days']} Days** | 💰 **Budget: ₹{ctx['budget']:,.0f}**\n"
                    f"🏨 **Hotel:** {trip['hotel']['name']} @ ₹{trip['hotel']['price_per_night']}/night\n\n"
                    "Your complete day-by-day plan is ready in the dashboard above. "
                    "You can ask me to modify it anytime!"
                )
                response = {
                    "message": msg,
                    "type": "trip_generated",
                    "trip": trip,
                    "suggestions": [
                        f"Show activities for Day 1",
                        f"What are the must-try foods in {ctx['destination']}?",
                        "Change hotel to budget option",
                        "Add adventure activities"
                    ]
                }
            else:
                response = {
                    "message": f"❌ {trip['message']}",
                    "type": "error",
                    "suggestions": [f"Plan a trip to {s}" for s in trip.get("suggestions", [])]
                }
            session.add_message("assistant", response["message"])
            return response

        # Ask for missing info
        missing = []
        if not ctx["destination"]: missing.append("destination")
        if not ctx["days"]: missing.append("number of days")
        if not ctx["budget"]: missing.append("budget (e.g., ₹15,000)")

        prompts = {
            "destination": "🗺️ Where would you like to travel? (Goa, Manali, Jaipur, Kerala, Udaipur, Shimla)",
            "number of days": f"📅 How many days is your trip to {ctx.get('destination', 'your destination')}?",
            "budget (e.g., ₹15,000)": f"💰 What's your total budget in rupees?"
        }
        ask = missing[0]
        response = {
            "message": prompts.get(ask, f"Could you tell me your {ask}?"),
            "type": "clarification",
            "missing": missing,
        }
        session.add_message("assistant", response["message"])
        return response

    # ── Info / RAG search ────────────────────────────────────────────────────
    if is_info_request or not is_plan_request:
        dest_name = ctx.get("destination") or extracted_dest or ""
        rag_results = store.search(user_message, top_k=5, destination_filter=dest_name or None)

        if rag_results:
            # Build a rich response from retrieved docs
            relevant = rag_results[:3]
            parts = []
            for doc in relevant:
                if doc["type"] == "place":
                    p = doc["raw"]
                    parts.append(
                        f"**{p['name']}** — {p.get('description', '')} "
                        f"(Rating: ⭐{p.get('rating', 'N/A')}, Cost: ₹{p.get('cost', 0)})"
                    )
                elif doc["type"] == "destination":
                    d = doc["raw"]
                    parts.append(
                        f"**{d['name']}**, {d['state']} — {d['description']}"
                    )
                elif doc["type"] == "tip":
                    parts.append(f"💡 {doc['raw']['tip']}")
                elif doc["type"] == "hotel":
                    h = doc["raw"]
                    parts.append(
                        f"**{h['name']}** ({h.get('tier', '')}) — ₹{h.get('price_per_night', 0)}/night, "
                        f"⭐{h.get('rating', 'N/A')}"
                    )

            answer = "Here's what I found:\n\n" + "\n\n".join(f"• {p}" for p in parts)

            response = {
                "message": answer,
                "type": "info",
                "suggestions": [
                    f"Plan a trip to {dest_name}" if dest_name else "Plan a trip to Goa",
                    "What's the best budget hotel?",
                    "What are must-try foods?",
                ]
            }
        else:
            response = {
                "message": (
                    "I have detailed information about Goa, Manali, Jaipur, Kerala, Udaipur, and Shimla. "
                    "Try asking something like:\n"
                    "• *\"Best beaches in Goa\"*\n"
                    "• *\"Budget hotels in Manali\"*\n"
                    "• *\"Plan a 5-day Kerala trip under ₹20,000\"*"
                ),
                "type": "fallback",
            }
        session.add_message("assistant", response["message"])
        return response
