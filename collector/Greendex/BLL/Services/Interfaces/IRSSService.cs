using Data.DTO;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace BLL.Services
{
    public interface IRSSService
    {
        Task<List<RSSResult>> GetNewsLinks(DateTime lastCheck);
    }
}