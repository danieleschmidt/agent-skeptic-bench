"""Scenarios API endpoints."""


from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db_session
from ..database.repositories import ScenarioRepository
from ..models import ScenarioCategory
from .schemas import ScenarioCreate, ScenarioResponse, ScenarioUpdate

router = APIRouter()


@router.get("/", response_model=list[ScenarioResponse])
async def list_scenarios(
    categories: list[str] | None = Query(None, description="Filter by categories"),
    limit: int | None = Query(None, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    search: str | None = Query(None, description="Search in name and description"),
    db: AsyncSession = Depends(get_db_session)
):
    """List scenarios with optional filtering."""
    repo = ScenarioRepository(db)

    # Parse categories
    parsed_categories = None
    if categories:
        try:
            parsed_categories = [ScenarioCategory(cat) for cat in categories]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid category: {e}")

    # Search or list scenarios
    if search:
        scenarios = await repo.search(search, parsed_categories)
    else:
        scenarios = await repo.get_all(parsed_categories, limit, offset)

    return [ScenarioResponse.from_record(scenario) for scenario in scenarios]


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(
    scenario_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific scenario by ID."""
    repo = ScenarioRepository(db)
    scenario = await repo.get_by_id(scenario_id)

    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return ScenarioResponse.from_record(scenario)


@router.post("/", response_model=ScenarioResponse, status_code=201)
async def create_scenario(
    scenario_data: ScenarioCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new scenario."""
    repo = ScenarioRepository(db)

    # Check if scenario with same ID already exists
    existing = await repo.get_by_id(scenario_data.id)
    if existing:
        raise HTTPException(status_code=409, detail="Scenario with this ID already exists")

    # Create domain model
    scenario = scenario_data.to_domain_model()

    # Save to database
    record = await repo.create(scenario)
    await repo.commit()

    return ScenarioResponse.from_record(record)


@router.put("/{scenario_id}", response_model=ScenarioResponse)
async def update_scenario(
    scenario_id: str,
    scenario_data: ScenarioUpdate,
    db: AsyncSession = Depends(get_db_session)
):
    """Update an existing scenario."""
    repo = ScenarioRepository(db)

    # Check if scenario exists
    existing = await repo.get_by_id(scenario_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Scenario not found")

    # Update fields
    updates = scenario_data.dict(exclude_unset=True)
    if "category" in updates:
        updates["category"] = updates["category"].value

    record = await repo.update(scenario_id, updates)
    await repo.commit()

    return ScenarioResponse.from_record(record)


@router.delete("/{scenario_id}", status_code=204)
async def delete_scenario(
    scenario_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Delete a scenario."""
    repo = ScenarioRepository(db)

    deleted = await repo.delete(scenario_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")

    await repo.commit()


@router.get("/categories/stats")
async def get_category_stats(
    db: AsyncSession = Depends(get_db_session)
):
    """Get scenario count by category."""
    repo = ScenarioRepository(db)
    stats = await repo.count_by_category()

    return {
        "categories": stats,
        "total": sum(stats.values())
    }


@router.get("/category/{category}", response_model=list[ScenarioResponse])
async def list_scenarios_by_category(
    category: str,
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """List scenarios in a specific category."""
    try:
        scenario_category = ScenarioCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    repo = ScenarioRepository(db)
    scenarios = await repo.get_by_category(scenario_category)

    # Apply pagination
    if offset:
        scenarios = scenarios[offset:]
    if limit:
        scenarios = scenarios[:limit]

    return [ScenarioResponse.from_record(scenario) for scenario in scenarios]
