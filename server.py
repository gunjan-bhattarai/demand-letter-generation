#!/usr/bin/env python3
"""
Legal Case Management MCP Server
Provides access to case information, documents, timelines, and financial summaries
"""

import asyncio
import json
from typing import Any, Optional, List, Dict
from datetime import datetime
import os

import asyncpg
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource, ServerCapabilities

# Database connection configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "database": os.getenv("DB_NAME", "legal_case_management"),
    "user": os.getenv("DB_USER", "username"),
    "password": os.getenv("DB_PASSWORD", "password")
}

class LegalCaseServer:
    def __init__(self):
        self.server = Server("legal-case-server")
        self.pool: Optional[asyncpg.Pool] = None
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup all the MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available tools"""
            return [
                Tool(
                    name="get_case_details",
                    description="Retrieve complete case information including all parties",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The case identifier"
                            }
                        },
                        "required": ["case_id"]
                    }
                ),
                Tool(
                    name="get_case_documents",
                    description="Get document file paths, optionally filtered by category",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The case identifier"
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional document category filter (medical, financial, correspondence, police_report)",
                                "enum": ["medical", "financial", "correspondence", "police_report"]
                            }
                        },
                        "required": ["case_id"]
                    }
                ),
                Tool(
                    name="get_case_timeline",
                    description="Retrieve chronological events for the case",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The case identifier"
                            },
                            "event_type": {
                                "type": "string",
                                "description": "Optional event type filter (accident, medical_treatment, expense, correspondence)",
                                "enum": ["accident", "medical_treatment", "expense", "correspondence"]
                            }
                        },
                        "required": ["case_id"]
                    }
                ),
                Tool(
                    name="get_financial_summary",
                    description="Calculate total medical expenses, lost wages, and other damages",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The case identifier"
                            }
                        },
                        "required": ["case_id"]
                    }
                ),
                Tool(
                    name="search_similar_cases",
                    description="Find precedent cases for reference based on case_type and keywords",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_type": {
                                "type": "string",
                                "description": "The type of case to search for"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords to search for in case summaries"
                            }
                        },
                        "required": ["case_type"]
                    }
                ),
                Tool(
                    name="get_party_details",
                    description="Get specific party information (plaintiff, defendant, witness, insurance_company)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The case identifier"
                            },
                            "party_type": {
                                "type": "string",
                                "description": "Type of party to retrieve",
                                "enum": ["plaintiff", "defendant", "witness", "insurance_company"]
                            }
                        },
                        "required": ["case_id", "party_type"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            if not self.pool:
                return [TextContent(type="text", text="Database connection not initialized")]
            
            try:
                if name == "get_case_details":
                    result = await self.get_case_details(arguments["case_id"])
                    
                elif name == "get_case_documents":
                    result = await self.get_case_documents(
                        arguments["case_id"],
                        arguments.get("category")
                    )
                    
                elif name == "get_case_timeline":
                    result = await self.get_case_timeline(
                        arguments["case_id"],
                        arguments.get("event_type")
                    )
                    
                elif name == "get_financial_summary":
                    result = await self.get_financial_summary(arguments["case_id"])
                    
                elif name == "search_similar_cases":
                    result = await self.search_similar_cases(
                        arguments["case_type"],
                        arguments.get("keywords", [])
                    )
                    
                elif name == "get_party_details":
                    result = await self.get_party_details(
                        arguments["case_id"],
                        arguments["party_type"]
                    )
                    
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(**DB_CONFIG)
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.pool:
            await self.pool.close()
    
    async def get_case_details(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete case information including all parties"""
        async with self.pool.acquire() as conn:
            # Get case info
            case_query = """
                SELECT case_id, case_type, date_filed, status, attorney_id, case_summary
                FROM cases
                WHERE case_id = $1
            """
            case_row = await conn.fetchrow(case_query, case_id)
            
            if not case_row:
                return None
            
            # Get all parties for this case
            parties_query = """
                SELECT party_type, name, contact_info, insurance_info
                FROM parties
                WHERE case_id = $1
            """
            parties_rows = await conn.fetch(parties_query, case_id)
            
            return {
                "case_id": case_row["case_id"],
                "case_type": case_row["case_type"],
                "date_filed": case_row["date_filed"].isoformat() if case_row["date_filed"] else None,
                "status": case_row["status"],
                "attorney_id": case_row["attorney_id"],
                "case_summary": case_row["case_summary"],
                "parties": [
                    {
                        "party_type": row["party_type"],
                        "name": row["name"],
                        "contact_info": row["contact_info"] or {},
                        "insurance_info": row["insurance_info"] or {}
                    }
                    for row in parties_rows
                ]
            }
    
    async def get_case_documents(self, case_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get document file paths, optionally filtered by category"""
        async with self.pool.acquire() as conn:
            if category:
                query = """
                    SELECT file_path, doc_category, upload_date, document_title, metadata
                    FROM documents
                    WHERE case_id = $1 AND doc_category = $2
                    ORDER BY upload_date DESC
                """
                rows = await conn.fetch(query, case_id, category)
            else:
                query = """
                    SELECT file_path, doc_category, upload_date, document_title, metadata
                    FROM documents
                    WHERE case_id = $1
                    ORDER BY upload_date DESC
                """
                rows = await conn.fetch(query, case_id)
            
            return [
                {
                    "file_path": row["file_path"],
                    "doc_category": row["doc_category"],
                    "upload_date": row["upload_date"].isoformat() if row["upload_date"] else None,
                    "document_title": row["document_title"],
                    "metadata": row["metadata"] or {}
                }
                for row in rows
            ]
    
    async def get_case_timeline(self, case_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve chronological events for the case"""
        async with self.pool.acquire() as conn:
            if event_type:
                query = """
                    SELECT event_date, event_type, description, amount
                    FROM case_events
                    WHERE case_id = $1 AND event_type = $2
                    ORDER BY event_date
                """
                rows = await conn.fetch(query, case_id, event_type)
            else:
                query = """
                    SELECT event_date, event_type, description, amount
                    FROM case_events
                    WHERE case_id = $1
                    ORDER BY event_date
                """
                rows = await conn.fetch(query, case_id)
            
            return [
                {
                    "event_date": row["event_date"].isoformat() if row["event_date"] else None,
                    "event_type": row["event_type"],
                    "description": row["description"],
                    "amount": float(row["amount"]) if row["amount"] else None
                }
                for row in rows
            ]
    
    async def get_financial_summary(self, case_id: str) -> Dict[str, float]:
        """Calculate total medical expenses, lost wages, and other damages"""
        async with self.pool.acquire() as conn:
            # Try to get medical expenses from document metadata first
            medical_docs_query = """
                SELECT metadata->>'total_expenses' as total_expenses
                FROM documents
                WHERE case_id = $1 AND doc_category = 'medical'
                LIMIT 1
            """
            medical_doc = await conn.fetchrow(medical_docs_query, case_id)
            
            if medical_doc and medical_doc["total_expenses"]:
                total_medical = float(medical_doc["total_expenses"])
            else:
                # Fallback to summing events
                medical_sum_query = """
                    SELECT COALESCE(SUM(amount), 0) as total
                    FROM case_events
                    WHERE case_id = $1
                    AND event_type IN ('medical_treatment', 'expense')
                    AND (
                        description ILIKE '%medical%' 
                        OR description ILIKE '%therapy%'
                        OR description ILIKE '%prescription%'
                        OR description ILIKE '%ambulance%'
                    )
                """
                medical_sum = await conn.fetchrow(medical_sum_query, case_id)
                total_medical = float(medical_sum["total"])
            
            # Get lost wages from financial documents
            wages_docs_query = """
                SELECT metadata->>'total_wage_loss' as wage_loss
                FROM documents
                WHERE case_id = $1 AND doc_category = 'financial'
                LIMIT 1
            """
            wages_doc = await conn.fetchrow(wages_docs_query, case_id)
            lost_wages = float(wages_doc["wage_loss"]) if wages_doc and wages_doc["wage_loss"] else 0.0
            
            # Calculate other damages
            other_expenses_query = """
                SELECT COALESCE(SUM(amount), 0) as total
                FROM case_events
                WHERE case_id = $1
                AND event_type = 'expense'
                AND NOT (
                    description ILIKE '%medical%'
                    OR description ILIKE '%therapy%'
                    OR description ILIKE '%prescription%'
                    OR description ILIKE '%ambulance%'
                )
            """
            other_expenses = await conn.fetchrow(other_expenses_query, case_id)
            
            # Property damage from correspondence
            property_damage_query = """
                SELECT COALESCE(SUM(amount), 0) as total
                FROM case_events
                WHERE case_id = $1
                AND event_type = 'correspondence'
                AND description ILIKE '%property damage%'
            """
            property_damage = await conn.fetchrow(property_damage_query, case_id)
            
            other_damages = float(other_expenses["total"]) + float(property_damage["total"])
            
            return {
                "total_medical_expenses": total_medical,
                "lost_wages": lost_wages,
                "other_damages": other_damages,
                "total_damages": total_medical + lost_wages + other_damages
            }
    
    async def search_similar_cases(self, case_type: str, keywords: List[str]) -> List[str]:
        """Find precedent cases for reference based on case_type and keywords"""
        async with self.pool.acquire() as conn:
            if keywords:
                # Build the WHERE clause for keywords
                keyword_conditions = " AND ".join([f"case_summary ILIKE ${i+2}" for i in range(len(keywords))])
                query = f"""
                    SELECT case_id
                    FROM cases
                    WHERE case_type = $1
                    AND ({keyword_conditions})
                """
                # Prepare parameters: case_type + keywords with % wildcards
                params = [case_type] + [f"%{kw}%" for kw in keywords]
                rows = await conn.fetch(query, *params)
            else:
                query = """
                    SELECT case_id
                    FROM cases
                    WHERE case_type = $1
                """
                rows = await conn.fetch(query, case_type)
            
            return [row["case_id"] for row in rows]
    
    async def get_party_details(self, case_id: str, party_type: str) -> List[Dict[str, Any]]:
        """Get specific party information (plaintiff, defendant, etc.)"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT name, contact_info, insurance_info
                FROM parties
                WHERE case_id = $1 AND party_type = $2
            """
            rows = await conn.fetch(query, case_id, party_type)
            
            return [
                {
                    "name": row["name"],
                    "contact_info": row["contact_info"] or {},
                    "insurance_info": row["insurance_info"] or {}
                }
                for row in rows
            ]
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="legal-case-server",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    )
                )
            )

async def main():
    """Main entry point"""
    server = LegalCaseServer()
    
    try:
        await server.initialize()
        await server.run()
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
