# Business Context — Engineering Knowledge Search

## The Gap in Pre6 Today

Pre6's current workflow: Engineer receives an RFQ → uploads CAD → runs Costing / Machinist / Lens / PPAP.

Every project starts from a blank slate. There is no memory of what was built before — no way to know if an identical bracket was quoted six months ago, what it cost, who made it, or what manufacturing issues came up.

## What This Feature Does

Before an engineer touches any Pre6 tool, they can ask: **"Have we built something like this before?"**

Upload a STEP file, type a description, or both. The system returns ranked similar past parts with their manufacturing context: process, cost, supplier, known issues, PPAP documents.

## How It Fits Into the Existing Pre6 Journey

```
Engineer receives RFQ
        ↓
[NEW] Search similar past parts
        ↓                        ↓
   Found similar             Nothing found
        ↓                        ↓
Reuse/adapt past          Start fresh with Pre6
process + cost data       Costing / Machinist / Lens
        ↓
Retrieved history becomes input context for Pre6 Costing & Machinist
("This part resembles Project X; X cost $420 on process Y with supplier Z")
```

The retrieved context is not just for the engineer's reference — it feeds directly into Pre6's AI tools as grounded prior knowledge, improving estimate accuracy.

## Business Value

**For the engineer:** Reuse past process decisions instead of rediscovering them. Reduce quoting time on repeat/similar work.

**For the company:** Surface institutional knowledge that's currently locked in project folders. Every engineer has access to what every other engineer has learned.

**For Pre6's product:** This creates compounding stickiness. Every new project a customer runs through Pre6 enriches their searchable knowledge base — making Pre6 more valuable over time and raising the switching cost. [Ex: If all my projects are on Adobe cloud, I wouldn't want to switch to canva or vice-versa]

## Future Extensions (in order of value)

1. **Duplicate detection** — flag near-identical designs before quoting, prevent redundant work
2. **Cost prediction** — "based on 5 similar past parts, estimated cost range is $X–$Y" with confidence
3. **Supplier recommendation** — "these suppliers machined similar parts at this tolerance"
4. **DFM warnings from history** — "Project X had a thin-wall failure at this feature; your part has the same geometry"
5. **Design recommendations** — nudge engineers toward manufacturing-friendly geometries based on past success/failure patterns
